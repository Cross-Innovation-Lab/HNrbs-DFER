from .S2DFE_real import S2DFE_real

def create_model(args):
    
    model = S2DFE_real(args)
    
    return model