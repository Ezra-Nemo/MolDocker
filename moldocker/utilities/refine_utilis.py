import os, io, lzma, time
import parmed, pickle, warnings
import numpy as np

from openmm.app import Modeller, HBonds, Simulation, NoCutoff
from openmm import LangevinMiddleIntegrator, Platform, Context, System, CustomExternalForce
from openmm.openmm import MinimizationReporter
from openff.toolkit import Molecule
from openff.toolkit.utils.toolkits import ToolkitRegistry, RDKitToolkitWrapper
from openmm.unit import (nanometer, kelvin, picoseconds,
                         femtoseconds, bar, kilocalorie_per_mole, kilojoule_per_mole)
from openmmforcefields.generators import SystemGenerator

from .pdbfixer.pdbfixer import PDBFixer
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

from .general_utilis import DockingTextExtractor, DockProgressTableView

warnings.filterwarnings("ignore", category=DeprecationWarning)
toolkit_registry = ToolkitRegistry([RDKitToolkitWrapper])

property_functions = {'mw'  : Descriptors.MolWt,
                      'hbd' : Descriptors.NumHDonors,
                      'hba' : Descriptors.NumHAcceptors,
                      'logp': Descriptors.MolLogP,
                      'tpsa': Descriptors.TPSA,
                      'rb'  : Descriptors.NumRotatableBonds,
                      'nor' : lambda mol: mol.GetRingInfo().NumRings(),
                      'fc'  : lambda mol: sum([atom.GetFormalCharge() for atom in mol.GetAtoms()]),
                      'nha' : Descriptors.HeavyAtomCount,
                      'mr'  : Descriptors.MolMR,
                      'na'  : lambda mol: mol.GetNumAtoms(),
                      'QED' : QED.qed}

chem_prop_to_full_name_map = {'mw'  : 'Molecular Weight'        ,
                              'hbd' : 'Hydrogen Bond Donors', 'hba' : 'Hydrogen Bond Acceptors' ,
                              'logp': 'LogP'                , 'tpsa': 'Topological Polar Surface Area',
                              'rb'  : 'Rotatable Bonds'     , 'nor' : 'Number of Rings'         ,
                              'fc'  : 'Formal Charge'       , 'nha' : 'Number of Heavy Atoms'   ,
                              'mr'  : 'Molar Refractivity'  , 'na'  : 'Number of Atoms'         ,
                              'QED' : 'QED'}

class ImplicitMinimizeComplex:
    def __init__(self, protein_pth: str, ligand_pth: str, ph: float,
                 is_docking,
                 minimize_tolerance: float=5.,
                 minimize_maxiter: int=0):
        self.ligand_partial_charge = 'mmff94'
        self.forcefield_kwargs = {
            "constraints"   : None,
            'soluteDielectric': 1.0,
            'solventDielectric': 80.0,}
        self.ph = ph
        self.is_docking = is_docking
        
        self.protein_forcefield = ['amber14/protein.ff14SB.xml', 'implicit/obc1.xml']
        # self.protein_forcefield = ['charmm36.xml', 'implicit/obc1.xml']
        self.ligand_forcefield = 'openff_unconstrained-2.2.1.offxml'
        self.temperature = 298 * kelvin
        self.friction = 1 / picoseconds
        self.pressure = 1 * bar
        self.minimize_tolerance = minimize_tolerance
        self.minimize_maxiter = minimize_maxiter
        self.protein_pth = protein_pth
        self.ligand_pth = ligand_pth
        
        try:
            self.platform = Platform.getPlatformByName('CUDA')
            # print('Using CUDA.')
        except:
            try:
                self.platform = Platform.getPlatformByName('OpenCL')
                # print('Using OpenCL.')
            except:
                self.platform = Platform.getPlatformByName('CPU')
                # print('Using CPU.')
    
    @staticmethod
    def pdb_fix_and_cleanup(pdb_pth: str, ph: float):
        fixer = PDBFixer(pdb_pth)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(ph)
        return fixer
    
    def setup_protein(self, protein_pth: str):
        protein = self.pdb_fix_and_cleanup(protein_pth, self.ph)
        return Modeller(protein.topology, protein.positions)
    
    def setup_ligand(self, ligand_pth: str):
        ligand = Molecule.from_file(ligand_pth, allow_undefined_stereo=True)
        ligand.assign_partial_charges(self.ligand_partial_charge, toolkit_registry=toolkit_registry)
        return ligand
    
    def setup_system_generator(self):
        return SystemGenerator(forcefields=self.protein_forcefield,
                               small_molecule_forcefield=self.ligand_forcefield,
                               forcefield_kwargs=self.forcefield_kwargs,
                               periodic_forcefield_kwargs={'nonbondedMethod': NoCutoff},
                               molecules=self.ligand)
    
    def constrain_backbone(self):
        force = CustomExternalForce('0.5 * k * ((x - x0)^2 + (y - y0)^2 + (z - z0)^2)')
        force.addPerParticleParameter('x0')
        force.addPerParticleParameter('y0')
        force.addPerParticleParameter('z0')
        force.addGlobalParameter('k', 1e5*kilojoule_per_mole/nanometer**2)
        
        positions = self.modeller.positions
        for atom in self.modeller.topology.atoms():
            if atom.residue.name != 'UNK':  # Find all non-ligand
                if atom.name in ['N', 'CA', 'C', 'O']:  # Set backbone to rigid
                    index = atom.index
                    position = positions[index]
                    force.addParticle(index, [position.x, position.y, position.z])
        
        self.system.addForce(force)
    
    def _perturb_ligand_positions(self, ligand_positions, max_displacement=0.1):
        positions = np.array(ligand_positions.value_in_unit(nanometer))
        displacements = np.random.uniform(-max_displacement, max_displacement, positions.shape)
        perturbed_positions = positions + displacements
        return perturbed_positions * nanometer
    
    def _simulate_annealing(self, initial_temp=500*kelvin, final_temp=298*kelvin,
                            total_steps=1000, steps_per_temp=100):
        integrator = self.simulation.integrator
        num_temp_steps = total_steps // steps_per_temp
        temp_schedule = np.linspace(initial_temp.value_in_unit(kelvin),
                                    final_temp.value_in_unit(kelvin), num_temp_steps)
        for temp in temp_schedule:
            integrator.setTemperature(temp * kelvin)
            self.simulation.step(steps_per_temp)
    
    def setup_simulation(self):
        ligand_topology = self.ligand.to_topology()
        # ligand_positions = self.perturb_ligand_positions(ligand_topology.get_positions().to_openmm())
        self.modeller.add(ligand_topology.to_openmm(), ligand_topology.get_positions().to_openmm())
        self.system: System = self.sys_generator.create_system(self.modeller.topology)
        self.constrain_backbone()
        integrator = LangevinMiddleIntegrator(self.temperature, self.friction, 1 * femtoseconds)
        self.simulation = Simulation(self.modeller.topology, self.system, integrator, self.platform)
        self.simulation.context.setPositions(self.modeller.positions)
        
    def minimize_energy(self):
        reporter = MinReporter(self.is_docking)
        if self.minimize_tolerance is None:
            self.simulation.minimizeEnergy(maxIterations=self.minimize_maxiter, reporter=reporter)
        else:
            self.simulation.minimizeEnergy(self.minimize_tolerance,
                                           self.minimize_maxiter, reporter=reporter)
            
    def split_complex(self):
        struct = parmed.openmm.load_topology(self.simulation.topology,
                                             self.system,
                                             self.simulation.context.getState(getPositions=True).getPositions())
        # struct.strip(':HOH,NA,CL')
        return struct, struct['!:UNK'], struct[':UNK'], self.ligand # complex, protein, ligand, ligand_mol
    
    def __call__(self):
        self.modeller = self.setup_protein(self.protein_pth)
        if not self.is_docking.value:
            yield None, None
            
        self.ligand = self.setup_ligand(self.ligand_pth)
        self.sys_generator = self.setup_system_generator()
        if not self.is_docking.value:
            yield None, None
        
        self.setup_simulation()
        curr_eng = self.simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        if not self.is_docking.value:
            yield None, None
            
        yield f'Simulation setup, current energy: {curr_eng:.4f} kJ/mol', None
        self.minimize_energy()
        curr_eng = self.simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        if not self.is_docking.value:
            yield None, None
            
        yield f'Energy minimized, current energy: {curr_eng:.4f} kJ/mol', self.split_complex()
        # yield f'Short Sim. &nbsp;Done, current energy: {curr_eng:.4f} kJ/mol', self.split_complex()
        # yield f'Energy minimized, current energy: {curr_eng:.4f} kJ/mol', None
        # self.simulation.step(500)
        # for _ in range(5):
        #     self.simulation.step(100)
        #     if not self.is_docking.value:
        #         yield None, None
        
        # curr_eng = self.simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        # yield f'Short Sim. &nbsp;Done, current energy: {curr_eng:.4f} kJ/mol', self.split_complex()

class CalculateBindingEnergy:
    def __init__(self,
                 complex_struct: parmed.structure.Structure,
                 protein_struct: parmed.structure.Structure,
                 ligand_struct : parmed.structure.Structure,
                 ligand: Molecule,
                 is_docking):
        self.complex = complex_struct
        self.protein = protein_struct
        self.ligand  = ligand_struct
        self.is_docking = is_docking
        self.forcefield_kwargs = {
            "constraints"   : None,
            'soluteDielectric': 1.0,
            'solventDielectric': 80.0,}
        self.implicit_solvent_system_generator = SystemGenerator(forcefields=['amber14-all.xml', 'implicit/obc1.xml'],
                                                                 small_molecule_forcefield='openff_unconstrained-2.2.1.offxml',
                                                                 molecules=[ligand],
                                                                 forcefield_kwargs=self.forcefield_kwargs,
                                                                 periodic_forcefield_kwargs={'nonbondedMethod': NoCutoff})
    
    def retrieve_potential_energy(self, struct: parmed.structure.Structure) -> float:
        system = self.implicit_solvent_system_generator.create_system(struct.topology)
        context = Context(system, LangevinMiddleIntegrator(298 * kelvin, 1 / picoseconds, 1 * femtoseconds))
        context.setPositions(struct.positions)
        eng = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilocalorie_per_mole)
        del context
        return eng
    
    def calculate_binding_energy(self) -> float:
        if not self.is_docking.value:
            return None
        protein_eng = self.retrieve_potential_energy(self.protein)
        
        if not self.is_docking.value:
            return None
        ligand_eng  = self.retrieve_potential_energy(self.ligand )
        
        if not self.is_docking.value:
            return None
        complex_eng = self.retrieve_potential_energy(self.complex)
        return complex_eng - protein_eng - ligand_eng

class MinReporter(MinimizationReporter):
    # Used to check if user forced stopped mid way, so we don't have to wait until the minimization is fully completed
    def __init__(self, is_docking):
        super().__init__()
        self.is_docking = is_docking
    
    def report(self, iteration, x, grad, args):
        return not self.is_docking

def process_protein_ligand_to_dict(protein, ligand, ligand_mol):
    protein_io = io.StringIO()
    protein.save(protein_io, format='pdb')
    ligand_io = io.StringIO()
    ligand.save(ligand_io, format='pdb')
    
    lig_mol = ligand_mol.to_rdkit()
    original_conf = lig_mol.GetConformer()
    pdb_mol = Chem.MolFromPDBBlock(ligand_io.getvalue(), removeHs=False)
    pdb_mol_conf = pdb_mol.GetConformer()
    for i in range(lig_mol.GetNumAtoms()):
        xyz_3d = pdb_mol_conf.GetAtomPosition(i)
        original_conf.SetAtomPosition(i, xyz_3d)
    prop = {chem_prop_to_full_name_map[k]: str(func(lig_mol)) for k, func in property_functions.items()}
    lig_mol = Chem.RemoveHs(lig_mol)
    
    ligand_pdb_str_list = Chem.MolToPDBBlock(pdb_mol).replace(' UNK ', ' UNL ').strip().split('\n')
    complex_pdb_str_list = protein_io.getvalue().strip().split('\n')[:-1]
    final_pos = int(complex_pdb_str_list[-1][6:11])
    for line in ligand_pdb_str_list:
        if   line.startswith('HETATM'):
            new_pos = int(line[6:11]) + final_pos
            line = line[:6] + f'{new_pos:>5}' + line[11:]
        elif line.startswith('CONECT'):
            conect_pos = []
            for i in range(6, 27, 5):
                pos = line[i:i+5].strip()
                if pos:
                    conect_pos.append(f'{int(pos) + final_pos:>5}')
                else:
                    break
            line = 'CONECT' + ''.join(conect_pos)
        complex_pdb_str_list.append(line)
    complex_str = '\n'.join(complex_pdb_str_list)
    output_dict = {'complex': complex_str,
                   'rdmol'  : lig_mol}
    return output_dict, prop

def save_dict_to_mdm(output_dict: dict, pth: str):
    with lzma.open(pth, 'wb') as f:
        pickle.dump(output_dict, f)

def minimize_complex(protein_pth: str, ligand_pth: str, output_pth: str, csv_pth: str,
                     text_extractor: DockingTextExtractor, log_tableview: DockProgressTableView):
    try:
        name = os.path.basename(os.path.dirname(protein_pth))
        read_next = False
        with open(ligand_pth) as f:
            for l in f:
                if read_next:
                    old_score = float(l)
                    break
                if l == '>  <VINA Energy>  (1) \n' or l == '>  <Old Score>  (1) \n':
                    read_next = True
        tik = time.perf_counter()
        minimize = ImplicitMinimizeComplex(protein_pth, ligand_pth)
        for progress, minimized_structs in minimize():
            if minimized_structs is None:
                text_extractor.update_text.emit(progress, False)
                log_tableview.update_progress_bar_by_add(name, 51/4)
            else:
                log_tableview.set_progress_bar_value(name, 51/4)
                complex, protein, ligand, ligand_mol = minimized_structs
                output_dict, prop = process_protein_ligand_to_dict(protein, ligand, ligand_mol)
                calculator = CalculateBindingEnergy(complex, protein, ligand, ligand_mol)
                binding_energy = calculator.calculate_binding_energy()
                log_tableview.set_progress_bar_value(name, 51)
                output_dict['binding_energy'] = binding_energy
                output_dict['old_score'] = old_score
                output_dict.update(prop)
                save_dict_to_mdm(output_dict, output_pth)
                tok = time.perf_counter() - tik
                with open(csv_pth, 'a') as f:
                    f.write(f'{name},{binding_energy},{old_score},{",".join(v for v in prop.values())}\n')
                text_extractor.update_text.emit(f'Binding Energy: {binding_energy:.4f} kcal/mol ({tok:.2f} sec)', False)
                log_tableview.update_progress_status(name, f'{binding_energy:.3f}')
    except Exception as e:
        empty = (len(property_functions) - 1) * ','
        with open(csv_pth, 'a') as f:
            f.write(f'{name},,{old_score},{empty}\n')
        text = str(e).replace('\n', '<br/>')
        text_extractor.update_text.emit(text, False)
        log_tableview.set_progress_bar_value(name, 51)
        log_tableview.update_progress_status(name, 'Failed')
        
def mp_minimize_complex(protein_pth: str, ligand_pth: str, output_pth: str, csv_pth: str,
                        ph: float, progress_queue, is_docking):
    if not os.path.isfile(ligand_pth) or not os.path.isfile(protein_pth):
        name = os.path.basename(os.path.dirname(protein_pth))
        empty = (len(property_functions) - 1) * ','
        with open(csv_pth, 'a') as f:
            f.write(f'{name},,,{empty}\n')
        text = f'{name} does not contain protein and/or ligand file!' + '<br/>'
        return text, name, 'Failed'
    try:
        name = os.path.basename(os.path.dirname(protein_pth))
        read_next = False
        with open(ligand_pth) as f:
            for l in f:
                if read_next:
                    old_score = float(l)
                    break
                if l == '>  <VINA Energy>  (1) \n' or l == '>  <Old Score>  (1) \n':
                    read_next = True
        final_strs = [f'Refining {name}...']
        tik = time.perf_counter()
        minimize = ImplicitMinimizeComplex(protein_pth, ligand_pth, ph, is_docking)
        for progress, minimized_structs in minimize():
            if progress is None:
                return f'User forced stopped "{name}".', name, 'Failed'
            if minimized_structs is None:
                progress_queue.put(name)
                final_strs.append(progress)
            else:
                progress_queue.put(name)
                final_strs.append(progress)
                complex, protein, ligand, ligand_mol = minimized_structs
                output_dict, prop = process_protein_ligand_to_dict(protein, ligand, ligand_mol)
                calculator = CalculateBindingEnergy(complex, protein, ligand, ligand_mol, is_docking)
                binding_energy = calculator.calculate_binding_energy()
                if binding_energy is None:
                    return f'User forced stopped "{name}".', name, 'Failed'
                progress_queue.put(name)
                output_dict['binding_energy'] = binding_energy
                output_dict['old_score'] = old_score
                output_dict.update(prop)
                save_dict_to_mdm(output_dict, output_pth)
                passed_time = time.perf_counter() - tik
                with open(csv_pth, 'a') as f:
                    f.write(f'{name},{binding_energy},{old_score},{",".join(v for v in prop.values())}\n')
                passed_min, passed_sec = divmod(passed_time, 60)
                final_strs.append(f'Binding Energy: {binding_energy:.4f} kcal/mol, Refining Time: {int(passed_min):02}:{int(passed_sec):02}')
                return '<br/>'.join(final_strs)+'<br/>', name, f'{binding_energy:.4f}'
    except Exception as e:
        empty = (len(property_functions) - 1) * ','
        with open(csv_pth, 'a') as f:
            f.write(f'{name},,{old_score},{empty}\n')
        text = f'Refining {name}...<br/>' + str(e).replace('\n', '<br/>') + '<br/>'
        return text, name, 'Failed'

def single_minimize_complex(name, input_dir, out_dir, csv_pth, ph, start_queue, progress_queue, is_docking):
    if not is_docking.value:
        return None, None, None
    output_mdm = os.path.join(out_dir, f'{name}_output.mdm')
    protein_pth = os.path.join(input_dir, 'protein.pdb')
    ligand_pth = os.path.join(input_dir, f'{name}.sdf')
    start_queue.put(name)
    return mp_minimize_complex(protein_pth, ligand_pth, output_mdm, csv_pth, ph, progress_queue, is_docking)
