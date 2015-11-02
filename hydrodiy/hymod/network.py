

class Connexion(object):

    def __init__(idx_upcomp, idx_downcomp, 
        idx_upcomp_output, idx_downcomp_input):

        # Index of upstream component
        self.idx_upcomp = idx_upcomp

        # Indexes of output from upstream component
        self.idx_upcomp_output = idx_upcomp_output

        # Index of downstream component
        self.idx_downcomp = idx_downcomp

        # Indexes of inputs sent to downstream component
        self.idx_downcomp = idx_downcomp_input


class Element(object):

    def __init__(self, id, is_conservative, is_sync):

        self.id = id

        # Are connexions conservative ? 
        self.is_conservative = is_conservative

        # Are connexions in sync within the same time step ?
        self.is_sync = is_sync


class Network(object):
    
    def __init__(self):
        

    def add_element(element):
