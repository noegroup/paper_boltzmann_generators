__author__ = 'noe'

import unittest
import keras
import numpy as np
from deep_boltzmann.models import ParticleDimer

class ParticleDimerTest(unittest.TestCase):

    def setUp(self):
        self.model = ParticleDimer()
        self.x0 = self.model.init_positions(0.7)
        self.x0 = self.x0.astype(np.float32)
        self.accuracy = 0.01

    def test_LJ_energy(self):
        E = self.model.LJ_energy(self.x0)
        E_tf = keras.backend.eval(self.model.LJ_energy_tf(self.x0))
        assert(E - E_tf < self.accuracy)

    def test_dimer_energy(self):
        E = self.model.dimer_energy(self.x0)
        E_tf = keras.backend.eval(self.model.dimer_energy_tf(self.x0))
        assert(E - E_tf < self.accuracy)

    def test_box_energy(self):
        E = self.model.box_energy(self.x0)
        E_tf = keras.backend.eval(self.model.box_energy_tf(self.x0))
        assert(E - E_tf < self.accuracy)

    def test_energy(self):
        E = self.model.energy(self.x0)
        E_tf = keras.backend.eval(self.model.energy_tf(self.x0))
        assert(E - E_tf < self.accuracy)


if __name__ == '__main__':
    unittest.main()
