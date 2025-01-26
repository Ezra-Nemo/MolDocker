import os, re, time, shutil
import platform, subprocess

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolAlign import CalcRMS
from openbabel import pybel
from multiprocessing import Manager, Queue

from concurrent.futures import ProcessPoolExecutor, wait

from PySide6.QtCore import Signal, QTimer, QObject, Slot, QThread

class DockingThread(QThread):
    def run(self):
        self.exec_()

### LeDock ###
def run_multiprocess_ledock(dock_exe: str, dock_setting_file: str, target_dok_file: str, procid_list, is_docking):
    cmd = [f'{dock_exe}', dock_setting_file]
    proc = subprocess.Popen(cmd,
                            stdin =subprocess.DEVNULL, 
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            start_new_session=True)
    procid_list.append(proc.pid)
    try:
        proc.wait(20)
    except subprocess.TimeoutExpired:
        while True:
            if not is_docking.value:
                break
            elif os.path.isfile(target_dok_file) and os.path.getsize(target_dok_file) > 0:
                break
            else:
                retcode = proc.poll()
                if retcode is not None:
                    break
            time.sleep(5)
    procid_list.remove(proc.pid)

def _write_ledock_settings_and_ligand_list(param_dict: dict, cp_ligand_pth: str, name: str):
    with open(param_dict['ligand_l'].format(lig_name=name), 'w') as f:
        f.write(cp_ligand_pth)
    
    with open(param_dict['setting_f'].format(lig_name=name), 'w') as f:
        f.write('RMSD\n1.0\n\n')
        f.write(f'Binding pocket\n{param_dict['x_min']:.3f} {param_dict['x_max']:.3f}\n')
        f.write(f'{param_dict['y_min']:.3f} {param_dict['y_max']:.3f}\n')
        f.write(f'{param_dict['z_min']:.3f} {param_dict['z_max']:.3f}\n\n')
        f.write(f'Number of binding poses\n{param_dict['poses']}\n\n')
        f.write(f'Receptor\n{param_dict['receptor']}\n\n')
        f.write(f'Ligands list\n{param_dict['ligand_l'].format(lig_name=name)}\n\n')
        f.write('END')

def _convert_ledock_to_sdf(before_docked_mol2_pth: str, dok_ligand_pth: str, eng_compiled: re.compile):
    # Why can't every program just output the same universal format? Why even call it .dok? It is nearly the same as PDB...
    # Not everyone knows how to program these stuff, so this is just going to make the general population "fear" this,
    # thinking these witchcrafts are inaccessible and thus not trying to understand and learn them.
    # WE NEED MORE PEOPLE LEARNING IN ORDER TO IMPROVE!
    # Need to recover original bond & stereocenter information since PDB format created by LeDock doesn't keep those.
    mol = Chem.MolFromMol2File(before_docked_mol2_pth, removeHs=False)
    if mol is None:
        ob_mol = next(pybel.readfile('mol2', before_docked_mol2_pth))
        molblock = ob_mol.write('sdf')
        mol = Chem.MolFromMolBlock(molblock, removeHs=False)
        if mol is None:
            return None
    
    with open(dok_ligand_pth) as f:
        docked_str = f.read()
    eng_list = []
    total_atoms = mol.GetNumAtoms()
    
    for i in re.finditer(eng_compiled, docked_str):
        mdl = i.group()
        eng = float(mdl.split('\n')[0].split('Score: ')[1].split('kcal/mol')[0])
        docked_mol = Chem.MolFromPDBBlock(mdl, removeHs=False)
        if docked_mol is None:
            # Fall back to openbabel if failed to read the pdb file
            pybel_mol = pybel.readstring('pdb', mdl)
            mol_block = pybel_mol.write('mol')
            docked_mol = Chem.MolFromMolBlock(mol_block, removeHs=False)
        if docked_mol is None:
            continue
        if docked_mol.GetNumAtoms() != total_atoms:
            # Shouldn't happen, but just in case
            continue
        
        conf_id = mol.AddConformer(Chem.Conformer(total_atoms), assignId=True)
        conformer = mol.GetConformer(conf_id)
        docked_conf = docked_mol.GetConformer()
        
        # use the original conformer as template, then set each atom coord. to the docked atom coord.
        for atom_idx in range(total_atoms):
            position = docked_conf.GetAtomPosition(atom_idx)
            conformer.SetAtomPosition(atom_idx, position)
        
        eng_list.append(eng)
    
    if not eng_list:
        return None
    
    mol.RemoveConformer(0)  # this is the original mol2 format conformer, remove it
    
    name = os.path.basename(dok_ligand_pth).rsplit('.dok')[0]
    out_sdf_pth = os.path.join(os.path.dirname(dok_ligand_pth), f'{name}_out.sdf')
    with Chem.SDWriter(out_sdf_pth) as writer:
        num_confs = mol.GetNumConformers()
        for conf_id in range(1, num_confs + 1):
            try:
                rmsd = CalcRMS(mol, mol, 1, conf_id)
            except:
                rmsd = 0.00
            mol.SetProp('Docking Metrics', f'Score: {eng_list[conf_id-1]} RMSD: 0.00 RMSD: {rmsd:.2f}')
            writer.write(mol, confId=conf_id)
    return eng_list

def single_ledock(ledock_exec: str, origin_lig: str, target_lig: str, param_dict: dict,
                  eng_compiled: re.compile, procid_list, queue, is_docking):
    shutil.copy(origin_lig, target_lig)
    name = os.path.basename(target_lig).rsplit('.', 1)[0]
    if not is_docking.value:
        return None, None, None
    _write_ledock_settings_and_ligand_list(param_dict, target_lig, name)
    docked_file = os.path.join(os.path.dirname(target_lig), name+'.dok')
    queue.put(name)
    
    tik = time.perf_counter()
    run_multiprocess_ledock(ledock_exec,
                            param_dict['setting_f'].format(lig_name=name),
                            docked_file,
                            procid_list,
                            is_docking)
    passed_time = time.perf_counter() - tik
    if not is_docking.value:
        # os.remove(target_lig)
        # os.remove(docked_file)
        # os.remove(param_dict['setting_f'].format(lig_name=name))
        # os.remove(param_dict['ligand_l'].format(lig_name=name))
        return f'Force stopped "{name}"', name, 'Failed'
    
    progress_csv = os.path.join(os.path.dirname(param_dict['setting_f']), 'dock_progress.csv')
    if os.path.isfile(docked_file):
        docked_eng = _convert_ledock_to_sdf(target_lig, docked_file, eng_compiled)
        os.remove(target_lig)
        os.remove(docked_file)
        os.remove(param_dict['setting_f'].format(lig_name=name))
        os.remove(param_dict['ligand_l'].format(lig_name=name))
        if docked_eng is None:
            with open(progress_csv, 'a+') as f:
                f.write(f'{name},\n')
            return f'Failed to convert {name}.<br/>', name, 'Failed'
        rank_len = len(str(len(docked_eng)))
        eng_pose_strs = [f'Rank {i:<{rank_len}}: {eng:.2f} kcal/mol' for i, eng in enumerate(docked_eng, start=1)]
        passed_min, passed_sec = divmod(passed_time, 60)
        text = (f'Name: {name}, Docking Time: {int(passed_min):02}:{int(passed_sec):02}<br/>Energies: <br/>'
                f'{"<br/>".join(eng_pose_strs)}<br/>')
        with open(progress_csv, 'a+') as f:
            f.write(f'{name},{docked_eng[0]:.2f}\n')
        return text, name, f'{docked_eng[0]:.2f}'
    else:
        os.remove(param_dict['setting_f'].format(lig_name=name))
        os.remove(param_dict['ligand_l'].format(lig_name=name))
        os.remove(target_lig)
        with open(progress_csv, 'a+') as f:
            f.write(f'{name},\n')
        return f'{name} Failed<br/>', name, 'Failed'

class MultiprocessLeDock(QObject):
    doneSignal = Signal(str, str, str) # docking result text, name, status text
    startSignal = Signal(str)
    finished = Signal(bool)
    canceled = Signal()
    withinCanceled = Signal()
    
    def __init__(self, ledock_exec: str, all_ligands: dict, param_dict: dict, concurrent_num: int):
        super().__init__()
        self.ledock_exec = ledock_exec
        self.all_ligands = all_ligands
        self.param_dict = param_dict
        self.ledock_eng_compiled = re.compile(r'REMARK Cluster.*+\n((.|\n)*?)END.*+')
        self.docked_dir = os.path.dirname(param_dict['ligand_l'])
        self.futures = []
        self.max_workers = concurrent_num
    
    def run(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_queue)
        self.timer.start(10)
        self.withinCanceled.connect(self.on_canceled)
        self.finished.connect(self.on_finished)
        self.finished.connect(self.timer.stop)
        QTimer.singleShot(0, self.run_docking)
    
    def run_docking(self):
        self.manager = Manager()
        self.procid_list = self.manager.list()
        self.queue = self.manager.Queue()
        self.is_docking = self.manager.Value('b', True)
        self.curr_docking = True
        self.executor = ProcessPoolExecutor(self.max_workers)
        
        for origin_pth, target_pth in self.all_ligands.items():
            future = self.executor.submit(
                single_ledock,
                self.ledock_exec,
                origin_pth,
                target_pth,
                self.param_dict,
                self.ledock_eng_compiled,
                self.procid_list,
                self.queue,
                self.is_docking
            )
            future.add_done_callback(self.process_future)
            self.futures.append(future)
            
    def process_future(self, future):
        try:
            result, name, status = future.result()
            if result is not None:
                self.doneSignal.emit(result, name, status)
        except Exception as e:
            pass
            
        if all(f.done() for f in self.futures):
            self.finished.emit(self.curr_docking)
    
    def check_queue(self):
        if not self.queue.empty():
            name = self.queue.get()
            self.startSignal.emit(name)
    
    @Slot()
    def stop(self):
        self.is_docking.value = False
        self.curr_docking = False
        self.wtihinCanceled.emit()
    
    @Slot()
    def on_canceled(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.manager.shutdown()
        self.timer.stop()
        self.canceled.emit()
    
    @Slot(bool)
    def on_finished(self, is_docking):
        self.executor.shutdown()
        self.manager.shutdown()

### VINA & Variants ###
def fetch_output(proc, name: str, queue: Queue, is_docking):
    while True:
        if not is_docking.value:
            proc.kill()
            break
        char = proc.stdout.read(1)
        if not char and proc.poll() is not None:
            break
        if char and char == '*':
            queue.put(name)

def run_vina(param_dict: dict, program_type: str, name: str, is_windows: bool, pid_list, queue, is_docking):
    cmd = [f'{param_dict['vina_exec']}',
           '--receptor'      , f"{param_dict['receptor_path']}"   ,
           '--ligand'        , f"{param_dict['ligand_path']}"     ,
           '--center_x'      , f"{param_dict['dock_center']['x']}",
           '--center_y'      , f"{param_dict['dock_center']['y']}",
           '--center_z'      , f"{param_dict['dock_center']['z']}",
           '--size_x'        , f"{param_dict['dock_width']['x']}" ,
           '--size_y'        , f"{param_dict['dock_width']['y']}" ,
           '--size_z'        , f"{param_dict['dock_width']['z']}" ,
           '--exhaustiveness', f"{param_dict['exhaustiveness']}"  ,
           '--num_modes'     , f"{param_dict['eval_poses']}"      ,
           '--out'           , f"{param_dict['docked_path']}"     ,
           '--energy_range'  , '10'                               ,]
    if 'flex_receptor_path' in param_dict:
        cmd += ['--flex', f"{param_dict['flex_receptor_path']}"]
        if program_type == 'smina':
            cmd += ['--out_flex', f"{param_dict['out_flex']}"]
    if program_type == 'smina':
        cmd += ['--addH', 'False']
    if not is_windows:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, start_new_session=True)
    else:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    pid_list.append(proc.pid)
    fetch_output(proc, name, queue, is_docking)
    proc.wait()
    pid_list.remove(proc.pid)

def fix_smina_sdf_file(out_pth: str, param_dict: dict,
                       smina_eng_re: re.compile, smina_flex_res_re: re.compile):
    # PDBQT format is now recommended! Meeko works a lot better than my code and 
    # pdbqt format as input to generate output pdbqt format does not result in incorrect valence.
    # bc pdbqt doesn't record bond information too.
    if os.path.getsize(out_pth) == 0:
        os.remove(out_pth)
        if 'out_flex' in param_dict:
            os.remove(param_dict['out_flex'])
        return 'Failed', None
    else:
        # Time to reconstruct the docked molecule using the original molecule! I hate programming :)
        original_mol = Chem.RemoveAllHs(next(Chem.SDMolSupplier(param_dict['ligand_path'], removeHs=False)))
        
        with open(out_pth) as f:
            sdf_str = f.read()
        eng_list = re.findall(smina_eng_re, sdf_str)
        
        # TODO: Remove the dependency of openbabel (try RDKit without sanitization)
        obabel_gen = pybel.readfile('sdf', out_pth)
        final_mols = []
        success_indices = []
        
        for idx, obabel_mol in enumerate(obabel_gen):
            obabel_mol.addh()
            pdb = obabel_mol.write('pdb')
            pdb = pdb.replace('ATOM  ', 'HETATM')
            r = []
            for l in pdb.split('\n'):
                # Can't keep the CONECT record, because some C can have 5 bonds according to openbabel within smina.
                # This is the main reason why rdkit can't read/sanitize some of the files, incorrect valence for A LOT of atoms.
                # I am converting it to PDB format because PDB format doesn't necessarily require bond information between HETATM,
                # so I don't have to deal with incorrect valence. However, this causes everything to be viewed as bonded by a single bond,
                # To deal with this, rdkit is used to "reconstruct" the original molecule by assigning bond order of original molecule
                # Sadly, positions of hydrogens are lost this way...
                if l.startswith('HETATM'):
                    line = l[:17] + 'UNL' + l[20:]
                    r.append(line)
                elif l.startswith('COMPND'):
                    r.append(l)
            pdb = '\n'.join(r)
            docked_mol = Chem.MolFromPDBBlock(pdb)
            if docked_mol is None:
                continue
            try:
                original_mol = Chem.RemoveAllHs(original_mol)
                docked_mol = AllChem.AssignBondOrdersFromTemplate(original_mol, docked_mol)
                AllChem.AssignStereochemistryFrom3D(docked_mol, replaceExistingTags=True)
                AllChem.AssignStereochemistry(docked_mol, force=True, cleanIt=True)
                # docked_mol = Chem.AddHs(docked_mol, addCoords=True)
            except Exception as e:
                continue
            final_mols.append(docked_mol)
            success_indices.append(idx)
        
        if not success_indices:
            return 'Failed', None
        
        # Replace original sdf file
        with Chem.SDWriter(out_pth) as writer:
            for idx, docked_mol in zip(success_indices, final_mols):
                docked_mol.SetProp('minimizedAffinity', f'{eng_list[idx]}')
                writer.write(docked_mol)
        
        # Finally, set the supplier to the new sdf file
        # Everything should be fixed by now
        # (beside some stereocenters, which could lead to RDKit UFF forcefield not being able 
        # to read the molecules to assign charges)
        supp = Chem.SDMolSupplier(out_pth)
        final_eng_list = [eng_list[idx] for idx in success_indices]
        
        if 'out_flex' in param_dict:
            with open(param_dict['out_flex']) as f:
                flex_pdb = f.read()
            all_flex_pos = [g[0] for idx, g in enumerate(re.findall(smina_flex_res_re, flex_pdb)) if idx in success_indices]
            final_mols = []
            for idx, m in enumerate(supp):
                m.SetProp('Flex Sidechains PDB', all_flex_pos[idx])
                final_mols.append(m)
            with Chem.SDWriter(out_pth) as writer:
                for mol in final_mols:
                    writer.write(mol)
            os.remove(param_dict['out_flex'])
        
        return 'Done', final_eng_list

def fix_smina_pdbqt_file(out_pth: str, param_dict: dict,
                         smina_flex_res_re: re.compile):
    # Recommeded over sdf format now!
    if os.path.getsize(out_pth) == 0:
        os.remove(out_pth)
        if 'out_flex' in param_dict:
            os.remove(param_dict['out_flex'])
        return 'Failed', None
    else:
        all_flex_pos = []
        eng_list = []
        if 'out_flex' in param_dict:
            with open(param_dict['out_flex']) as f:
                flex_pdb = f.read()
            all_flex_pos = [g[0] for g in re.findall(smina_flex_res_re, flex_pdb)]
            os.remove(param_dict['out_flex'])
        flex_idx = 0
        
        with open(out_pth) as f:
            pdbqt_lines = []
            for line in f:
                if line.startswith('REMARK minimizedAffinity'):
                    affinity = line.split()[-1]
                    eng_list.append(affinity)
                    new_line = f'REMARK VINA RESULT: {affinity} 0.00 0.00\n'
                    pdbqt_lines.append(new_line)
                elif line.startswith('TORSDOF') and all_flex_pos:
                    pdbqt_lines.append('BEGIN_RES\n')
                    pdbqt_lines.append(all_flex_pos[flex_idx])
                    pdbqt_lines.append('END_RES\n')
                    flex_idx += 1
                else:
                    pdbqt_lines.append(line)
        
        with open(out_pth, 'w') as f:
            f.writelines(pdbqt_lines)
        
        return 'Done', eng_list

def single_vina_dock(param_dict: dict, lig_pth: str, out_pth: str, program_type: str,
                     compiled_re_dict: dict,
                     start_que: Queue, progress_que: Queue,
                     is_windows, pid_list, is_docking):
    param_dict['ligand_path'] = lig_pth
    param_dict['docked_path'] = out_pth
    name = os.path.basename(lig_pth).rsplit('.', 1)[0]
    if not is_docking.value:
        return None, None, None
    if program_type == 'smina' and 'flex_receptor_path' in param_dict:
        # just make it pdb since I have to process it to pdb in the future anyway
        tmp_flex_pth = os.path.join(os.path.dirname(out_pth), name+'_flex.pdb')
        param_dict['out_flex'] = tmp_flex_pth
    start_que.put(name) # Send signal for which ligand is starting so I can update progress bar
    
    tik = time.perf_counter()
    run_vina(param_dict, program_type, name, is_windows, pid_list, progress_que, is_docking)
    passed_time = time.perf_counter() - tik
    if not is_docking.value:
        if program_type == 'smina':
            if os.path.isfile(out_pth):
                os.remove(out_pth)
                if 'out_flex' in param_dict:
                    os.remove(tmp_flex_pth)
        return f'Force stopped "{name}".', name, 'Failed'
    
    progress_csv = os.path.join(os.path.dirname(out_pth), 'cache_files', 'dock_progress.csv')
    if program_type == 'smina':
        smina_re = compiled_re_dict[program_type]
        if out_pth.endswith('.sdf'):
            status, eng_list = fix_smina_sdf_file(out_pth, param_dict, smina_re[0], smina_re[1])
        elif out_pth.endswith('.pdbqt'):
            status, eng_list = fix_smina_pdbqt_file(out_pth, param_dict, smina_re[1])
        if status == 'Failed':
            if os.path.isfile(out_pth):
                docking_result_text = f'Failed to process output file of "{name}".<br/>'
                os.remove(out_pth)
                if 'out_flex' in param_dict:
                    os.remove(tmp_flex_pth)
            else:
                docking_result_text = f'Failed to dock "{name}".<br/>'
        else:
            status = 'Done'
            rank_len = len(str(len(eng_list)))
            eng_pose_strs = [f'Rank {i:<{rank_len}}: {eng} kcal/mol' for i, eng in enumerate(eng_list, start=1)]
            passed_min, passed_sec = divmod(passed_time, 60)
            docking_result_text = (f'Name: {name}, Docking Time: {int(passed_min):02}:{int(passed_sec):02}<br/>Energies: <br/>'
                                   f'{"<br/>".join(eng_pose_strs)}<br/>')
    elif program_type in ['AutoDock VINA', 'qvina2', 'qvinaw']:
        # Make sure if VINA is sucessful or not, not sure if the fetch_output decision is good enough
        if not os.path.isfile(out_pth):
            status = 'Failed'
            docking_result_text = f'Failed to dock "{name}".<br/>'
        elif os.path.getsize(out_pth) == 0:
            os.remove(out_pth)
            status = 'Failed'
            docking_result_text = f'Failed to dock "{name}".<br/>'
        else:
            status = 'Done'
            with open(out_pth) as f:
                docked_str = f.read()
            vina_re = compiled_re_dict[program_type][0]
            eng_list = re.findall(vina_re, docked_str)
            rank_len = len(str(len(eng_list)))
            passed_min, passed_sec = divmod(passed_time, 60)
            eng_pose_strs = [f'Rank {i:<{rank_len}}: {eng} kcal/mol' for i, eng in enumerate(eng_list, start=1)]
            docking_result_text = (f'Name: {name}, Docking Time: {int(passed_min):02}:{int(passed_sec):02}<br/>Energies: <br/>'
                                   f'{"<br/>".join(eng_pose_strs)}<br/>')
    
    if status == 'Done':
        with open(progress_csv, 'a+') as file:
            file.write(f'{name},{float(eng_list[0]):.2f}\n')
    elif status == 'Failed':
        with open(progress_csv, 'a+') as file:
            file.write(f'{name},\n')
    
    final_status = f'{float(eng_list[0]):.2f}' if status == 'Done' else status
    
    return docking_result_text, name, final_status

class MultiprocessVINADock(QObject):
    doneSignal = Signal(str, str, str) # docking result text, name, status text (score / Failed)
    progressSignal = Signal(str)
    startSignal = Signal(str)
    finished = Signal(bool)
    canceled = Signal()
    withinCanceled = Signal()
    
    def __init__(self, lig_out_dict: dict, param_dict: dict, program_type: str, concurrent_num: int):
        super().__init__()
        self.lig_out_dict = lig_out_dict
        self.param_dict = param_dict
        self.program_type = program_type
        self.re_compiled_dict = {'smina': [],
                                 'AutoDock VINA': [],
                                 'qvina2': [],
                                 'qvinaw': []}
        if program_type == 'smina':
            self.re_compiled_dict[program_type].append(re.compile(r'>\s*<minimizedAffinity>(?:\s*\(\d+\))?\s?\n\s?(-?\d+\.\d+)'))    # sdf energy
            # self.re_compiled_dict[program_type].append(re.compile(r'REMARK minimizedAffinity (-?\d+\.?\d*)'))    # pdbqt energy
            self.re_compiled_dict[program_type].append(re.compile(r'AUTHOR.*+\n((.|\n)*?)CONECT.*\n'))   # Flex res
        elif program_type in ['AutoDock VINA', 'qvina2', 'qvinaw']:
            self.re_compiled_dict[program_type].append(re.compile(r'REMARK VINA RESULT:\s+(-?\d+\.\d+)'))   # eng
        self.docked_dir = os.path.dirname(next(iter(lig_out_dict.values())))
        self.futures = []
        self.max_workers = concurrent_num
        self.is_windows = platform.system() == 'Windows'
    
    def run(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_queues)
        self.timer.start(1)
        self.withinCanceled.connect(self.on_canceled)
        self.finished.connect(self.on_finished)
        self.finished.connect(self.timer.stop)
        QTimer.singleShot(0, self.run_docking)
        
    def run_docking(self):
        self.manager = Manager()
        self.procid_list = self.manager.list()
        self.start_queue = self.manager.Queue()
        self.progress_queue = self.manager.Queue()
        self.curr_docking = True
        self.is_docking = self.manager.Value('b', True)
        self.executor = ProcessPoolExecutor(self.max_workers)
        
        for lig_pth, out_pth in self.lig_out_dict.items():
            future = self.executor.submit(
                single_vina_dock,
                self.param_dict,
                lig_pth,
                out_pth,
                self.program_type,
                self.re_compiled_dict,
                self.start_queue,
                self.progress_queue,
                self.is_windows,
                self.procid_list,
                self.is_docking,
            )
            future.add_done_callback(self.process_future)
            self.futures.append(future)
            
    def process_future(self, future):
        try:
            result, name, status = future.result()
            if result is not None:
                self.doneSignal.emit(result, name, status)
        except Exception as e:
            pass
            
        if all(f.done() for f in self.futures):
            self.finished.emit(self.curr_docking)
    
    def check_queues(self):
        # Check if "name" start docking
        if not self.start_queue.empty():
            name = self.start_queue.get()
            self.startSignal.emit(name)
        
        # Update the docking progress of "name"
        if not self.progress_queue.empty():
            name = self.progress_queue.get()
            self.progressSignal.emit(name)
    
    @Slot()
    def stop(self):
        self.is_docking.value = False
        self.curr_docking = False
        self.withinCanceled.emit()
        
    @Slot()
    def on_canceled(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.manager.shutdown()
        self.timer.stop()
        self.canceled.emit()
        
    @Slot(bool)
    def on_finished(self, is_docking):
        self.executor.shutdown()
        self.manager.shutdown()

### Refine ###
class MultiprocessRefine(QObject):
    doneSignal = Signal(str, str, str) # docking result text, name, success?
    progressSignal = Signal(str, float)
    startSignal = Signal(str, str)
    finished = Signal(bool)
    canceled = Signal()
    withinCanceled = Signal()
    
    def __init__(self, all_complex_map: dict, out_dir: str, minimize_csv: str, concurrent_num: int, ph: float):
        super().__init__()
        self.all_complex_map = all_complex_map
        self.out_dir = out_dir
        self.minimize_csv = minimize_csv
        self.futures = []
        self.max_workers = concurrent_num
        self.ph = ph
    
    def run(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_queues)
        self.timer.start(10)
        self.withinCanceled.connect(self.on_canceled)
        self.finished.connect(self.on_finished)
        QTimer.singleShot(0, self.run_refining)
        
    def run_refining(self):
        from .refine_utilis import single_minimize_complex
        
        self.manager = Manager()
        self.start_queue = self.manager.Queue()
        self.progress_queue = self.manager.Queue()
        self.is_docking = self.manager.Value('b', True)
        self.curr_docking = True
        self.executor = ProcessPoolExecutor(self.max_workers)
        
        for name, input_dir in self.all_complex_map.items():
            future = self.executor.submit(
                single_minimize_complex,
                name,
                input_dir,
                self.out_dir,
                self.minimize_csv,
                self.ph,
                self.start_queue,
                self.progress_queue,
                self.is_docking,
            )
            future.add_done_callback(self.process_future)
            self.futures.append(future)
            
    def process_future(self, future):
        try:
            result, name, status = future.result()
            if result is not None:
                self.doneSignal.emit(result, name, status)
        except Exception as e:
            pass
            
        if all(f.done() for f in self.futures):
            self.finished.emit(self.curr_docking)
    
    def check_queues(self):
        # Check if "name" start docking
        if not self.start_queue.empty():
            name = self.start_queue.get()
            self.startSignal.emit(name, 'Refining...')
        
        # Update the docking progress of "name"
        if not self.progress_queue.empty():
            name = self.progress_queue.get()
            self.progressSignal.emit(name, 51 / 3)
    
    @Slot()
    def stop(self):
        self.is_docking.value = False
        self.curr_docking = False
        self.withinCanceled.emit()
    
    @Slot()
    def on_canceled(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.manager.shutdown()
        self.timer.stop()
        self.canceled.emit()
    
    @Slot()
    def on_finished(self):
        self.executor.shutdown()
        self.manager.shutdown()
        self.timer.stop()

class MultiprocessRefineFunc:
    def __init__(self, all_complex_map: dict, out_dir: str, minimize_csv: str, concurrent_num: int, ph: float):
        self.all_complex_map = all_complex_map
        self.out_dir = out_dir
        self.minimize_csv = minimize_csv
        self.futures = []
        self.max_workers = concurrent_num
        self.ph = ph
        
    def run(self):
        from .refine_utilis import single_minimize_complex
        self.manager = Manager()
        self.start_queue = self.manager.Queue()
        self.progress_queue = self.manager.Queue()
        self.is_docking = self.manager.Value('b', True)
        
        self.total_tasks = len(self.all_complex_map)
        self.progress_bar = tqdm(total=self.total_tasks, desc="Processing", unit="task")
        
        with ProcessPoolExecutor(self.max_workers) as executor:
            for name, input_dir in self.all_complex_map.items():
                future = executor.submit(
                    single_minimize_complex,
                    name,
                    input_dir,
                    self.out_dir,
                    self.minimize_csv,
                    self.ph,
                    self.start_queue,
                    self.progress_queue,
                    self.is_docking,
                )
                future.add_done_callback(self.process_future)
                self.futures.append(future)
            wait(self.futures)
        self.manager.shutdown()
        print('Done')
        
    def process_future(self, future):
        try:
            result, name, status = future.result()
        except Exception as e:
            pass
        self.progress_bar.update(1)