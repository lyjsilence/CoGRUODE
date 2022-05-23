
def config_LSST(model_name, args):
    params = {}

    params['dropout'] = args.dropout
    params['input_size'] = 6
    params['hidden_size'] = args.n_dim * params['input_size']
    params['bias'] = True
    params['num_class'] = 14
    params['batch_size'] = args.batch_size

    if model_name in ['CoGRUODE_HV', 'CoGRUODE_HM']:
        params['n_dim'] = args.n_dim
        params['init_hidden_state'] = True
        params['minimal'] = False
        params['solver'] = 'euler'
        params['memory'] = args.memory

    elif model_name in ['ComGRUODE', 'ComGRUODE_test']:
        params['n_dim'] = args.n_dim
        params['init_hidden_state'] = True
        params['minimal'] = True
        params['solver'] = 'euler'
        params['memory'] = args.memory

    elif model_name == 'GRUODE':
        params['minimal'] = False
        params['solver'] = 'euler'

    elif model_name == 'mGRUODE':
        params['minimal'] = True
        params['solver'] = 'euler'

    elif model_name == 'ODELSTM':
        params['cell_size'] = params['hidden_size']
        params['solver'] = 'euler'

    elif model_name == 'ODERNN':
        params['solver'] = 'euler'

    elif model_name == 'GRU-D':
        pass

    elif model_name == 'GRU_delta_t':
        pass

    elif model_name == 'CTGRU':
        params['scale'] = 5

    else:
        raise ModuleNotFoundError

    return params

def config_Activity(model_name, args):
    params = {}

    params['dropout'] = args.dropout
    params['input_size'] = 12
    params['hidden_size'] = args.n_dim * params['input_size']
    params['bias'] = True
    params['num_class'] = 7
    params['batch_size'] = args.batch_size

    if model_name in ['CoGRUODE_HV', 'CoGRUODE_HM']:
        params['n_dim'] = args.n_dim
        params['init_hidden_state'] = True
        params['minimal'] = False
        params['solver'] = 'euler'
        params['memory'] = args.memory

    elif model_name in ['ComGRUODE', 'ComGRUODE_test']:
        params['n_dim'] = args.n_dim
        params['init_hidden_state'] = True
        params['minimal'] = True
        params['solver'] = 'euler'
        params['memory'] = args.memory

    elif model_name in ['GRUODE', 'GRUODE_test']:
        params['minimal'] = False
        params['solver'] = 'euler'

    elif model_name == 'mGRUODE':
        params['minimal'] = True
        params['solver'] = 'euler'

    elif model_name == 'ODELSTM':
        params['cell_size'] = params['hidden_size']
        params['solver'] = 'euler'

    elif model_name == 'ODERNN':
        params['solver'] = 'euler'

    elif model_name == 'GRU-D':
        pass

    elif model_name == 'GRU_delta_t':
        pass

    elif model_name == 'CTGRU':
        params['scale'] = 5

    else:
        raise ModuleNotFoundError

    return params

def config_PhysioNet(model_name, args):
    params = {}

    params['dropout'] = args.dropout
    params['input_size'] = 37
    params['hidden_size'] = args.n_dim * params['input_size']
    params['bias'] = True
    params['num_class'] = 2
    params['batch_size'] = args.batch_size

    if model_name in ['CoGRUODE_HV', 'CoGRUODE_HM']:
        params['n_dim'] = args.n_dim
        params['init_hidden_state'] = True
        params['minimal'] = False
        params['solver'] = 'euler'
        params['memory'] = args.memory

    elif model_name in ['ComGRUODE_HV', 'ComGRUODE_HM']:
        params['n_dim'] = args.n_dim
        params['init_hidden_state'] = True
        params['minimal'] = True
        params['solver'] = 'euler'
        params['memory'] = args.memory

    elif model_name == 'GRUODE':
        params['minimal'] = False
        params['solver'] = 'euler'

    elif model_name == 'mGRUODE':
        params['minimal'] = True
        params['solver'] = 'euler'

    elif model_name == 'ODELSTM':
        params['cell_size'] = params['hidden_size']
        params['solver'] = 'euler'

    elif model_name == 'ODERNN':
        params['solver'] = 'euler'

    elif model_name == 'GRU-D':
        pass

    elif model_name == 'GRU_delta_t':
        pass

    elif model_name == 'CTGRU':
        params['scale'] = 5

    else:
        raise ModuleNotFoundError

    return params


def config(model_name, args):
    if args.dataset == 'LSST':
        params = config_LSST(model_name, args)

    elif args.dataset == 'Activity':
        params = config_Activity(model_name, args)

    elif args.dataset == 'PhysioNet':
        params = config_PhysioNet(model_name, args)

    return params

