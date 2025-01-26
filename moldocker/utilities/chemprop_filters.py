
def retrieve_filter_params(filter_name: str):
    if filter_name == 'Lipinski’s Filter':
        config = {'mw'  : [('≤', 500)],
                  'hbd' : [('≤', 5)],
                  'hba' : [('≤', 10)],
                  'logp': [('≤', 5)],
                  'tpsa': [()],
                  'rb'  : [()],
                  'nor' : [()],
                  'fc'  : [()],
                  'nha' : [()],
                  'mr'  : [()],
                  'na'  : [()]}
    elif filter_name == 'Veber Filter':
        config = {'mw'  : [()],
                  'hbd' : [()],
                  'hba' : [()],
                  'logp': [()],
                  'tpsa': [('≤', 140)],
                  'rb'  : [('≤', 10)],
                  'nor' : [()],
                  'fc'  : [()],
                  'nha' : [()],
                  'mr'  : [()],
                  'na'  : [()]}
    elif filter_name == 'Egan Filter':
        config = {'mw'  : [()],
                  'hbd' : [()],
                  'hba' : [()],
                  'logp': [('≤', 5.88)],
                  'tpsa': [('≤', 131.6)],
                  'rb'  : [()],
                  'nor' : [()],
                  'fc'  : [()],
                  'nha' : [()],
                  'mr'  : [()],
                  'na'  : [()]}
    elif filter_name == 'Palm Filter':
        config = {'mw'  : [()],
                  'hbd' : [()],
                  'hba' : [()],
                  'logp': [()],
                  'tpsa': [('≤', 140)],
                  'rb'  : [()],
                  'nor' : [()],
                  'fc'  : [()],
                  'nha' : [()],
                  'mr'  : [()],
                  'na'  : [()]}
    elif filter_name == 'REOS Filter':
        config = {'mw'  : [('>', 200), ('<', 500)],
                  'hbd' : [('<', 5)],
                  'hba' : [('<', 10)],
                  'logp': [('>', -5), ('<', 5)],
                  'tpsa': [()],
                  'rb'  : [('<', 8)],
                  'nor' : [()],
                  'fc'  : [('>', -2), ('<', 2)],
                  'nha' : [('>', 15), ('<', 50)],
                  'mr'  : [()],
                  'na'  : [()]}
    elif filter_name == 'Ghose Filter':
        config = {'mw'  : [('>', 160), ('<', 480)],
                  'hbd' : [('<', 5)],
                  'hba' : [('<', 10)],
                  'logp': [('>', -0.4), ('<', 5.6)],
                  'tpsa': [()],
                  'rb'  : [()],
                  'nor' : [()],
                  'fc'  : [()],
                  'nha' : [()],
                  'mr'  : [('>', 40), ('<', 130)],
                  'na'  : [('>', 20), ('<', 70)]}
    elif filter_name == 'Lead-like Filter':
        config = {'mw'  : [('≤', 300)],
                  'hbd' : [('≤', 3)],
                  'hba' : [('≤', 3)],
                  'logp': [('≤', 3)],
                  'tpsa': [()],
                  'rb'  : [()],
                  'nor' : [()],
                  'fc'  : [()],
                  'nha' : [()],
                  'mr'  : [()],
                  'na'  : [()]}
    elif filter_name == 'Van der Waterbeemd Filter':
        config = {'mw'  : [('≤', 450)],
                  'hbd' : [()],
                  'hba' : [()],
                  'logp': [()],
                  'tpsa': [('≤', 90)],
                  'rb'  : [()],
                  'nor' : [()],
                  'fc'  : [()],
                  'nha' : [()],
                  'mr'  : [()],
                  'na'  : [()]}
    elif filter_name == 'Murcko Filter':
        config = {'mw'  : [('≥', 200), ('≤', 400)],
                  'hbd' : [('≤', 3)],
                  'hba' : [('≤', 4)],
                  'logp': [('≤', 5.2)],
                  'tpsa': [()],
                  'rb'  : [('≤', 7)],
                  'nor' : [()],
                  'fc'  : [()],
                  'nha' : [()],
                  'mr'  : [()],
                  'na'  : [()]}
    elif filter_name == 'PPI Filter':
        config = {'mw'  : [('≥', 400)],
                  'hbd' : [()],
                  'hba' : [('≥', 4)],
                  'logp': [('≥', 4)],
                  'tpsa': [()],
                  'rb'  : [()],
                  'nor' : [('≥', 4)],
                  'fc'  : [()],
                  'nha' : [()],
                  'mr'  : [()],
                  'na'  : [()]}
    else:
        config = None
    return config