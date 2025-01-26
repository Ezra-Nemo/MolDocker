import re, os, io
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import AllChem
from shapely.geometry import Polygon

from .lowner_john_ellipse import welzl

aa_chain_pos_compiled = re.compile(r'......\s*\d+\s*(\w+)\s+([A-Z]{3}).([a-zA-Z0-9])\s*(-?\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)')
hetatm_conect_compiled = re.compile(r'(HETATM|CONECT).*\n')

class TwoWayMappingDict(dict):
    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

bond_type_map = {Chem.rdchem.BondType.SINGLE:   'Single Bond',
                 Chem.rdchem.BondType.DOUBLE:   'Double Bond',
                 Chem.rdchem.BondType.TRIPLE:   'Triple Bond',
                 Chem.rdchem.BondType.AROMATIC: 'Aromatic Bond',
                 'other'                      : 'Other',
                 }

interact_color_map = {'hydrogen bond'        : 'rgb(100, 149, 237)',
                      'hydrophobic contact'  : 'rgb( 86, 101, 115)',
                      'halogen bond'         : 'rgb(159, 226, 191)',
                      'ionic interaction'    : 'rgb(222,  49,  99)',
                      'cation-pi interaction': 'rgb(255, 127,  80)',
                      'pi-pi stacking'       : 'rgb(165, 105, 189)',}

atom_color_palette = {
     'H' : 'rgb(  0,   0,   0)',   # Hydrogen
     'C' : 'rgb(  0,   0,   0)',   # Carbon
     'N' : 'rgb(  0,   0, 255)',   # Nitrogen
     'O' : 'rgb(255,   0,   0)',   # Oxygen
     'F' : 'rgb( 51, 204, 204)',   # Fluorine
     'P' : 'rgb(255, 128,   0)',   # Phosphorus
     'S' : 'rgb(204, 204,   0)',   # Sulfur
     'Cl': 'rgb(  0, 204,   0)',   # Chlorine
     'Br': 'rgb(128,  77,  26)',   # Bromine
     'I' : 'rgb(161,  31, 240)',   # Iodine
}

def calc_offset(x12, y12, offset=0.05, shorten=None):
    dx = x12[:, 1] - x12[:, 0]
    dy = y12[:, 1] - y12[:, 0]
    dist = (dx ** 2 + dy ** 2) ** 0.5
    
    if shorten is not None:
        factor = shorten / dist
        x12[:, 0] = x12[:, 0] + factor * dx
        y12[:, 0] = y12[:, 0] + factor * dy
        x12[:, 1] = x12[:, 1] - factor * dx
        y12[:, 1] = y12[:, 1] - factor * dy
        
        dx = x12[:, 1] - x12[:, 0]
        dy = y12[:, 1] - y12[:, 0]
        dist = (dx ** 2 + dy ** 2) ** 0.5
    
    ox = offset * (-dy) / dist
    oy = offset * ( dx) / dist
    final_xa, final_ya, final_xb, final_yb = np.split(np.full((x12.shape[0] * 4, 3), None), 4, 0)
    final_xa[:, 0] = x12[:, 0] + ox
    final_xa[:, 1] = x12[:, 1] + ox
    final_ya[:, 0] = y12[:, 0] + oy
    final_ya[:, 1] = y12[:, 1] + oy
    
    final_xb[:, 0] = x12[:, 0] - ox
    final_xb[:, 1] = x12[:, 1] - ox
    final_yb[:, 0] = y12[:, 0] - oy
    final_yb[:, 1] = y12[:, 1] - oy
    return final_xa.flatten(), final_ya.flatten(), final_xb.flatten(), final_yb.flatten()

def create_ellipse(ellipse, num_pts, endpoint=True):
    # c = (x, y) is the center, a and b are the major and minor radii, and t is the rotation angle.
    c, a, b, t = ellipse
    rot_mat = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    theta = np.linspace(0, 2 * np.pi, num_pts, endpoint=endpoint)
    z = np.column_stack((a * np.cos(theta), b * np.sin(theta)))
    x = rot_mat.dot(z.T).T + c
    return x

def add_mol_to_fig(fig: go.Figure, mol: Chem.rdchem.Mol, idx: bool,
                   pro_atom_num_map: dict, pro_lig_interact_map: dict,
                   lig_coord_map=None, prev_ellipse=None):
    AllChem.Compute2DCoords(mol)
    conformer = mol.GetConformer()
    coords = np.array(conformer.GetPositions()).round(5)[:, :2]
    center_x, center_y = (coords.max(0) + coords.min(0)) / 2
    coords[:, 0] -= center_x
    coords[:, 1] -= center_y
    
    atom_coords_order = []
    for i, atom in enumerate(mol.GetAtoms()):
        sym = atom.GetSymbol()
        i = 1
        while f'{sym}{i}' in atom_coords_order:
            i += 1
        atom_coords_order.append(f'{sym}{i}')
    
    if idx != 0:
        all_keys = []
        for l in Chem.MolToPDBBlock(mol).split('\n'):
            if l.startswith('ATOM'):
                res, chain, pos = l[17:20].strip(), l[21], l[22:26].strip()
                key = f'[{res}]{pos}:{chain}'
                if key not in all_keys:
                    all_keys.append(key)
        pro_indices = [atom_coords_order.index(pro_atom_num_map[k][f'{k}.{pro_atom}']) 
                       for k in all_keys 
                       for pro_atom, _, _, _ in pro_lig_interact_map[k]]
        lig_coords = np.array([lig_coord_map[lig_atom] 
                               for k in all_keys 
                               for _, lig_atom, _, _ in pro_lig_interact_map[k]])
        
        anchor_res = coords[pro_indices].mean(axis=0)
        coords_shifted = coords - anchor_res
        anchor_lig = np.linalg.norm(lig_coords.mean(axis=0))
        
        all_prev_polygon = [Polygon(create_ellipse(ell, 50, True)) for ell in prev_ellipse]
        
        start_shift, shift_step = 1.2, 0.25
        coords_shifted += anchor_lig * start_shift
        shifted_center = np.zeros((2,))
        try:
            tmp_ellipse = list(welzl(coords_shifted))
        except:
            print('Ellipse failed')
            xy_max = coords_shifted.max(0)
            xy_min = coords_shifted.min(0)
            xy_cen = coords_shifted.mean(0)
            tmp_ellipse = [xy_cen, xy_max[0]-xy_min[0], xy_max[1]-xy_min[1], 0.]
        tmp_ellipse[1] += 0.2
        tmp_ellipse[2] += 0.2
        aa_polygon = Polygon(create_ellipse(tmp_ellipse, 50, True))
        intersect_area = [drawn_polygon.intersection(aa_polygon).area > 0 for drawn_polygon in all_prev_polygon]
        for e_i, a in enumerate(intersect_area):
            if a:
                shifted_center += (tmp_ellipse[0] - prev_ellipse[e_i][0])
        
        if sum(shifted_center):
            shifted_center = shifted_center / np.linalg.norm(shifted_center)
            find_best_shift = 0.2
            while True:
                shifted_anchor_lig = shifted_center * find_best_shift
                angles = np.linspace(0, 2 * np.pi, 15)
                correct_angle_idx = []
                correct_angle_dist = []
                for angle_idx, angle in enumerate(angles):
                    rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                                        [np.sin(angle),  np.cos(angle)],])
                    new_coords = coords_shifted @ rot_mat.T
                    new_coords += shifted_anchor_lig
                    try:
                        tmp_ellipse = list(welzl(new_coords))
                        tmp_ellipse[1] += 0.2
                        tmp_ellipse[2] += 0.2
                        aa_polygon = Polygon(create_ellipse(tmp_ellipse, 50, True))
                        intersect_area = [drawn_polygon.intersection(aa_polygon).area > 0 for drawn_polygon in all_prev_polygon]
                        if not sum(intersect_area):
                            correct_angle_idx.append(angle_idx)
                            correct_angle_dist.append(np.linalg.norm(new_coords[pro_indices].mean(0)-prev_ellipse[0][0]))
                        else:
                            shifted_center = np.zeros((2,))
                            for e_i, a in enumerate(intersect_area):
                                if a:
                                    shifted_center += (tmp_ellipse[0] - prev_ellipse[e_i][0])
                    except:
                        pass
                if correct_angle_idx:
                    angle = angles[correct_angle_idx[np.argmin(correct_angle_dist)]]
                    break
                find_best_shift += shift_step
            rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle),  np.cos(angle)],])
            coords = coords_shifted @ rot_mat.T
            coords += shifted_anchor_lig
        else:
            coords = coords_shifted
    
    try:
        ellipse = list(welzl(coords))
    except:
        xy_max = coords.max(0)
        xy_min = coords.min(0)
        xy_cen = coords.mean(0)
        ellipse = [xy_cen, xy_max[0]-xy_min[0], xy_max[1]-xy_min[1], 0.]
    ellipse[1] += 0.2
    ellipse[2] += 0.2
    prev_ellipse.append(ellipse)
    
    symbols_coords_map = {}
    for i, atom in enumerate(mol.GetAtoms()):
        sym = atom.GetSymbol()
        if sym not in symbols_coords_map:
            symbols_coords_map[sym] = [[], []]
        symbols_coords_map[sym][0].append(coords[i, 0])
        symbols_coords_map[sym][1].append(coords[i, 1])
        
    bond_type_edge_map = {Chem.rdchem.BondType.SINGLE  : [[], []],
                          Chem.rdchem.BondType.DOUBLE  : [[], []],
                          Chem.rdchem.BondType.TRIPLE  : [[], []],
                          Chem.rdchem.BondType.AROMATIC: [[], []],
                          'other'                      : [[], []],}
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        if bond_type not in bond_type_edge_map:
            bond_type = 'other'
        bond_type_edge_map[bond_type][0].extend([coords[i, 0], coords[j, 0], None])
        bond_type_edge_map[bond_type][1].extend([coords[i, 1], coords[j, 1], None])
    
    aromatic_centers = []
    
    for ring_group in mol.GetRingInfo().AtomRings():
        x, y = np.split(coords[list(ring_group)], 2, -1)
        center = [x.mean(), y.mean()]
        edge_midpoint = [x[:2].sum() / 2, y[:2].sum() / 2]
        max_radius = float(((np.array(center) - edge_midpoint) ** 2).sum() ** 0.5)
        aromatic_centers.append(center + [max_radius])
    
    ring_radius = 0.8
    for n, edge_x_y in bond_type_edge_map.items():
        edge_x, edge_y = edge_x_y
        b_t = bond_type_map[n]
        if idx == 0:
            color = 'rgb( 22, 160, 133)'
        else:
            color = 'black'
        if b_t in ['Single Bond', 'Other']:
            fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
                                    hoverinfo='none',
                                    name=b_t,
                                    showlegend=False,
                                    line={'width': 2, 'color': color},
                                    legendgroup='Bond', legendgrouptitle={'text': 'Bond'}))
        elif b_t == 'Double Bond':
            x = np.array(edge_x).reshape(-1, 3)[:, :2]
            y = np.array(edge_y).reshape(-1, 3)[:, :2]
            final_xa, final_ya, final_xb, final_yb = calc_offset(x, y, 0.06)
            fig.add_trace(go.Scatter(x=final_xa, y=final_ya,
                                    hoverinfo='none',
                                    mode='lines',
                                    line={'width': 1.5, 'color': color},
                                    showlegend=False,
                                    legendgroup='Bond', legendgrouptitle={'text': 'Bond'}))
            fig.add_trace(go.Scatter(x=final_xb, y=final_yb,
                                    hoverinfo='none',
                                    mode='lines',
                                    name=b_t,
                                    showlegend=False,
                                    line={'width': 1.5, 'color': color},
                                    legendgroup='Bond', legendgrouptitle={'text': 'Bond'}))
        elif b_t == 'Aromatic Bond':
            x = np.array(edge_x).reshape(-1, 3)[:, :2]
            y = np.array(edge_y).reshape(-1, 3)[:, :2]
            final_xa, final_ya, final_xb, final_yb = calc_offset(x, y, 0.05)
            fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
                                    hoverinfo='none',
                                    name=b_t,
                                    showlegend=False,
                                    line={'width': 2, 'color': color},
                                    legendgroup='Bond', legendgrouptitle={'text': 'Bond'}))
            for xyr_center in aromatic_centers:
                x, y, r = xyr_center
                fig.add_shape(type='circle', xref='x', yref='y',
                            x0=x-r*ring_radius, x1=x+r*ring_radius,
                            y0=y-r*ring_radius, y1=y+r*ring_radius,
                            line={'color': color, 'width': 1, 'dash': 'dot'})
        elif b_t == 'Triple Bond':
            x = np.array(edge_x).reshape(-1, 3)[:, :2]
            y = np.array(edge_y).reshape(-1, 3)[:, :2]
            final_xa, final_ya, final_xb, final_yb = calc_offset(x, y, 0.1, shorten=0.3)
            fig.add_trace(go.Scatter(x=final_xa, y=final_ya,
                                    hoverinfo='none',
                                    mode='lines',
                                    line={'width': 1, 'color': color},
                                    showlegend=False,
                                    legendgroup='Bond', legendgrouptitle={'text': 'Bond'}))
            fig.add_trace(go.Scatter(x=final_xb, y=final_yb,
                                    hoverinfo='none',
                                    mode='lines',
                                    line={'width': 1, 'color': color},
                                    showlegend=False,
                                    legendgroup='Bond', legendgrouptitle={'text': 'Bond'}))
            fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
                                    hoverinfo='none',
                                    mode='lines',
                                    name=b_t,
                                    line={'width': 1.5, 'color': color},
                                    showlegend=False,
                                    legendgroup='Bond', legendgrouptitle={'text': 'Bond'}))
    
    atom_coord_map = {}
    for sym, xy in symbols_coords_map.items():
        x, y = xy
        interaction_coords_dist = []
        if idx == 0:
            hov_texts = [f'{sym}{i}' for i in range(1, len(x) + 1)]
        else:
            hov_texts = []
            key_idx = 0
            sym_i = 1
            curr_k = all_keys[key_idx]
            curr_interact = pro_lig_interact_map[curr_k]
            proatom_unique = set([_[0] for _ in curr_interact])
            for i in range(1, len(x) + 1):
                if f'{sym}{sym_i}' not in pro_atom_num_map[curr_k]:
                    sym_i = 1
                    key_idx += 1
                    curr_k = all_keys[key_idx]
                    curr_interact = pro_lig_interact_map[curr_k]
                    proatom_unique = set([_[0] for _ in curr_interact])
                curr_text = pro_atom_num_map[curr_k][f'{sym}{sym_i}']
                curr_atom = curr_text.split('.')[-1]
                if curr_atom in proatom_unique:
                    for proatom_ligatom_dist_type in curr_interact:
                        proatom, ligatom, dist, inter_type = proatom_ligatom_dist_type
                        if curr_atom == proatom:
                            x_coord = [float(x[i-1]), lig_coord_map[ligatom][0]]
                            y_coord = [float(y[i-1]), lig_coord_map[ligatom][1]]
                            interaction_coords_dist.append((x_coord, y_coord, dist, inter_type, curr_text, ligatom))
                hov_texts.append(curr_text)
                sym_i += 1
        for s, xx, yy in zip(hov_texts, x, y):
            atom_coord_map[s.upper()] = np.array((float(xx), float(yy)))
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
                                marker={'color': atom_color_palette.get(sym, 'rgb(0, 0, 0)'),
                                        'line': {'color': 'black', 'width': 1},
                                        'size': 10},
                                name=sym, hovertemplate='%{customdata}<extra></extra>',
                                customdata=hov_texts,
                                legendgroup='Atom', legendgrouptitle={'text': 'Atom'},
                                showlegend=False))
        for x_y_d_t in interaction_coords_dist:
            x, y, d, t, pro, lig = x_y_d_t
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                                     line={'width': 3, 'color': interact_color_map[t], 'dash': '3, 2'},
                                     name=f'{pro} <-> {lig} ({d} Å)', visible='legendonly'))
    
    # ellipse_coords = create_ellipse(ellipse, 50, True)
    # fig.add_trace(go.Scatter(x=ellipse_coords[:, 0], y=ellipse_coords[:, 1],
    #                          showlegend=False, line={'color': 'black'}, hoverinfo='none'))
    
    if idx == 0:
        return fig, atom_coord_map, prev_ellipse
    return fig, prev_ellipse

def create_ligplot_figure(complex_pdb: str, interact_df: pd.DataFrame):
    if os.path.isfile(complex_pdb):
        with open(complex_pdb) as f:
            complex_pdb = f.read()
    pro_atom_num_map, extracted_pdb, used_regex = {}, '', []
    pro_atom_coords_map, pro_lig_interact_map = {}, {}
    for idx, row in interact_df.iterrows():
        protein, ligand, dist, interact_type = row['atom1'], row['atom2'], row['distance'], row['type']
        if protein.startswith('[UNL]'):
            protein, ligand = ligand, protein
        aa_pos, chain_atom = protein.split(':')
        aa, pos = aa_pos[1:].split(']')
        chain, atom = chain_atom.split('.')
        if f'{aa_pos}:{chain}' not in pro_atom_coords_map:
            pro_atom_coords_map[f'{aa_pos}:{chain}'] = []
            pro_lig_interact_map[f'{aa_pos}:{chain}'] = []
        appended_tuple = (atom, ligand.split('.')[-1], dist, interact_type)
        if appended_tuple not in pro_lig_interact_map[f'{aa_pos}:{chain}']:
            pro_lig_interact_map[f'{aa_pos}:{chain}'].append(appended_tuple)
        regex = rf'ATOM..\s*\d+\s*\w+\s+{aa}.{chain}\s*{pos}.*\n'
        if regex not in used_regex:
            aa_pdb_str = ''.join(re.findall(regex, complex_pdb))
            atom_name_map = TwoWayMappingDict()
            for l in aa_pdb_str.split('\n'):
                if l:
                    atom_type = l[-1]
                    if atom_type != 'H':
                        i = 1
                        while f'{atom_type}{i}' in atom_name_map:
                            i += 1
                        atom_name_map[f'{atom_type}{i}'] = f'{aa_pos}:{chain}.{l[12:15].strip()}'
                        pro_atom_coords_map[f'{aa_pos}:{chain}'].append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
            pro_atom_coords_map[f'{aa_pos}:{chain}'] = np.array(pro_atom_coords_map[f'{aa_pos}:{chain}'])
            pro_atom_num_map[f'{aa_pos}:{chain}'] = atom_name_map
            extracted_pdb += aa_pdb_str
            used_regex.append(regex)
            
    ligand_pdb, lig_atom_coords = '', []
    for match in re.finditer(hetatm_conect_compiled, complex_pdb):
        l = match.group(0)
        ligand_pdb += l
        if not l.startswith('CONECT'):
            lig_atom_coords.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
    lig_atom_coords = np.array(lig_atom_coords)
    
    extracted_pdb = ligand_pdb + extracted_pdb
    
    obmol = pybel.readstring('pdb', extracted_pdb)
    obmol.removeh()
    pdb_str = obmol.write('pdb')
    mol = Chem.MolFromPDBBlock(pdb_str)
    all_mols = Chem.GetMolFrags(mol, asMols=True)
    prev_ellipse = []
    fig = go.Figure()
    
    for unique_interact in interact_df['type'].unique():
        fig.add_trace(go.Bar(x=[0], y=[0], marker_color=interact_color_map[unique_interact],
                            name=unique_interact,
                            legendgroup='interactions group',
                            legendgrouptitle={'text': 'Interactions'}))
        
    for idx, mol in enumerate(all_mols):
        if idx == 0:
            fig, lig_coord_map, prev_ellipse = add_mol_to_fig(fig, mol, idx, pro_atom_num_map, pro_lig_interact_map,
                                                              prev_ellipse=prev_ellipse)
        else:
            fig, prev_ellipse = add_mol_to_fig(fig, mol, idx, pro_atom_num_map, pro_lig_interact_map,
                                               lig_coord_map, prev_ellipse)
            
    fig.update_layout(xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False, 'title': None},
                      yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False, 'title': None, 'scaleanchor': 'x'},
                      margin={'l': 0, 'r': 0, 't': 20, 'b': 0},
                      template='plotly_white')
    
    return fig
