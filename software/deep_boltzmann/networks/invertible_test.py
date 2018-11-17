import unittest
import numpy as np
from deep_boltzmann.networks.invertible import create_NICERNet, create_RealNVPNet

class InversionTest(unittest.TestCase):

    def setUp(self):
        self.accuracy = 1e-4
        class EnergyModelDummy(object):
            def __init__(self):
                self.dim=10
        self.nicer_net_unscaled = create_NICERNet(EnergyModelDummy(), nlayers=2, scaled=False)
        self.nicer_net_scaled = create_NICERNet(EnergyModelDummy(), nlayers=2, scaled=True)
        self.realnvp_net = create_RealNVPNet(EnergyModelDummy(), nlayers=2)

    def test_NICER_forward(self):
        x0 = np.random.randn(2, 10)
        for net in [self.nicer_net_unscaled, self.nicer_net_scaled, self.realnvp_net]:
            xr = net.transform_zx(net.transform_xz(x0))
            err = np.max(np.abs(x0 - xr))
            print(err)
            assert(err < self.accuracy)

    def test_NICER_backward(self):
        z0 = np.random.randn(2, 10)
        for net in [self.nicer_net_unscaled, self.nicer_net_scaled, self.realnvp_net]:
            zr = net.transform_xz(net.transform_zx(z0))
            err = np.max(np.abs(z0 - zr))
            print(err)
            assert(err < self.accuracy)



if __name__ == '__main__':
    unittest.main()
