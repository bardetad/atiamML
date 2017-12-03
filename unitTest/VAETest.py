import unittest
import sys
# Add the src folder path to the sys.path list
sys.path.append('../src')

from VAE import VAE

class TestVAE(unittest.TestCase):

    def test_good_VAE(self):
        X_dim = 513
        Z_dim = 6
        IOh_dims_Enc = [X_dim, 128, Z_dim]
        IOh_dims_Dec = [Z_dim, 128, X_dim]
        NL_types_Enc = ['relu']
        NL_types_Dec = ['relu', 'sigmoid']
        model = VAE(X_dim, Z_dim, IOh_dims_Enc, IOh_dims_Dec, NL_types_Enc, NL_types_Dec)
        self.assertTrue(model.created)

    def test_wrong_EncoderStructure(self):
        X_dim = 513
        Z_dim = 6
        IOh_dims_Enc = [X_dim, Z_dim]
        IOh_dims_Dec = [Z_dim, 128, X_dim]
        NL_types_Enc = ['relu']
        NL_types_Dec = ['relu', 'sigmoid']
        model = VAE(X_dim, Z_dim, IOh_dims_Enc, IOh_dims_Dec, NL_types_Enc, NL_types_Dec)
        self.assertFalse(model.created)

    def test_wrong_DecoderStructure(self):
        X_dim = 513
        Z_dim = 6
        IOh_dims_Enc = [X_dim, 128, Z_dim]
        IOh_dims_Dec = [X_dim]
        NL_types_Enc = ['relu']
        NL_types_Dec = ['relu', 'sigmoid']
        model = VAE(X_dim, Z_dim, IOh_dims_Enc, IOh_dims_Dec, NL_types_Enc, NL_types_Dec)
        self.assertFalse(model.created)

    def test_wrong_EncoderNLFunctionsNb(self):
        X_dim = 513
        Z_dim = 6
        IOh_dims_Enc = [X_dim, 128, Z_dim]
        IOh_dims_Dec = [Z_dim, 128, X_dim]
        NL_types_Enc = ['relu', 'relu']
        NL_types_Dec = ['relu', 'sigmoid']
        model = VAE(X_dim, Z_dim, IOh_dims_Enc, IOh_dims_Dec, NL_types_Enc, NL_types_Dec)
        self.assertFalse(model.created)

    def test_wrong_DecoderNLFunctionsNb(self):
        X_dim = 513
        Z_dim = 6
        IOh_dims_Enc = [X_dim, 128, Z_dim]
        IOh_dims_Dec = [Z_dim, 128, X_dim]
        NL_types_Enc = ['relu']
        NL_types_Dec = ['relu']
        model = VAE(X_dim, Z_dim, IOh_dims_Enc, IOh_dims_Dec, NL_types_Enc, NL_types_Dec)
        self.assertFalse(model.created)

    def test_wrong_EncoderNLfunctionsSyntax(self):
        X_dim = 513
        Z_dim = 6
        IOh_dims_Enc = [X_dim, 128, Z_dim]
        IOh_dims_Dec = [Z_dim, 128, X_dim]
        NL_types_Enc = ['reLu']
        NL_types_Dec = ['relu', 'sigmoid']
        model = VAE(X_dim, Z_dim, IOh_dims_Enc, IOh_dims_Dec, NL_types_Enc, NL_types_Dec)
        self.assertFalse(model.created)

    def test_wrong_DecoderNLfunctionsSyntax(self):
        X_dim = 513
        Z_dim = 6
        IOh_dims_Enc = [X_dim, 128, Z_dim]
        IOh_dims_Dec = [Z_dim, 128, X_dim]
        NL_types_Enc = ['relu']
        NL_types_Dec = ['relu', 'sigmoide']
        model = VAE(X_dim, Z_dim, IOh_dims_Enc, IOh_dims_Dec, NL_types_Enc, NL_types_Dec)
        self.assertFalse(model.created)

suiteEncoder = unittest.TestLoader().loadTestsFromTestCase(TestVAE)
print "\n\n------------------- VAE Test Suite -------------------\n"
unittest.TextTestRunner(verbosity=2).run(suiteEncoder)