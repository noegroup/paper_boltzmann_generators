import numpy as np
import tensorflow as tf

class MM(object):

    def __init__(self, toppar, align=10.0, align_atoms=[0,1,2]):
        self.toppar = toppar
        self.dim = 3*toppar.natoms
        self.natoms = toppar.natoms
        self.atom_indices = np.arange(self.dim).reshape(self.natoms, 3).astype(np.int32)
        self.align = align
        self.align_atoms = np.array(align_atoms, dtype=np.int32)

    def dist(self, x1, x2):
        d = x2-x1
        d2 = np.sum(d*d, axis=2)
        return np.sqrt(d2)

    def dist_tf(self, x1, x2):
        d = x2-x1
        d2 = tf.reduce_sum(d*d, axis=2)
        return tf.sqrt(d2)

    def angle(self, x1, x2, x3):
        ba = x1 - x2
        ba /= np.linalg.norm(ba, axis=2, keepdims=True)
        bc = x3 - x2
        bc /= np.linalg.norm(bc, axis=2, keepdims=True)
        cosine_angle = np.sum(ba*bc, axis=2)
        angle = np.degrees(np.arccos(cosine_angle))
        return angle

    def angle_tf(self, x1, x2, x3):
        ba = x1 - x2
        ba /= tf.norm(ba, axis=2, keepdims=True)
        bc = x3 - x2
        bc /= tf.norm(bc, axis=2, keepdims=True)
        cosine_angle = tf.reduce_sum(ba*bc, axis=2)
        angle = np.float32(180.0 / np.pi) * tf.acos(cosine_angle)
        return angle

    def torsion(self, x1, x2, x3, x4):
        """Praxeolitic formula
        1 sqrt, 1 cross product"""
        b0 = -1.0*(x2 - x1)
        b1 = x3 - x2
        b2 = x4 - x3
        # normalize b1 so that it does not influence magnitude of vector
        # rejections that come next
        b1 /= np.linalg.norm(b1, axis=2, keepdims=True)

        # vector rejections
        # v = projection of b0 onto plane perpendicular to b1
        #   = b0 minus component that aligns with b1
        # w = projection of b2 onto plane perpendicular to b1
        #   = b2 minus component that aligns with b1
        v = b0 - np.sum(b0*b1, axis=2, keepdims=True) * b1
        w = b2 - np.sum(b2*b1, axis=2, keepdims=True) * b1

        # angle between v and w in a plane is the torsion angle
        # v and w may not be normalized but that's fine since tan is y/x
        x = np.sum(v*w, axis=2)
        b1xv = np.cross(b1, v, axisa=2, axisb=2)
        y = np.sum(b1xv*w, axis=2)
        return np.degrees(np.arctan2(y, x))

    def torsion_tf(self, x1, x2, x3, x4):
        """Praxeolitic formula
        1 sqrt, 1 cross product"""
        b0 = -1.0*(x2 - x1)
        b1 = x3 - x2
        b2 = x4 - x3
        # normalize b1 so that it does not influence magnitude of vector
        # rejections that come next
        b1 /= tf.norm(b1, axis=2, keepdims=True)

        # vector rejections
        # v = projection of b0 onto plane perpendicular to b1
        #   = b0 minus component that aligns with b1
        # w = projection of b2 onto plane perpendicular to b1
        #   = b2 minus component that aligns with b1
        v = b0 - tf.reduce_sum(b0*b1, axis=2, keepdims=True) * b1
        w = b2 - tf.reduce_sum(b2*b1, axis=2, keepdims=True) * b1

        # angle between v and w in a plane is the torsion angle
        # v and w may not be normalized but that's fine since tan is y/x
        x = tf.reduce_sum(v*w, axis=2)
        b1xv = tf.cross(b1, v)
        y = tf.reduce_sum(b1xv*w, axis=2)
        return np.float32(180.0 / np.pi) * tf.atan2(y, x)

    def bondlengths(self, x):
        dists = self.dist(x[:, self.atom_indices[self.toppar.bond_indices[:, 0]]],
                          x[:, self.atom_indices[self.toppar.bond_indices[:, 1]]])
        return dists

    def nb_dists(self, x):
        dists = self.dist(x[:, self.atom_indices[self.toppar.nonbonded_indices[:, 0]]],
                          x[:, self.atom_indices[self.toppar.nonbonded_indices[:, 1]]])
        return dists

    def nb14_dists(self, x):
        dists = self.dist(x[:, self.atom_indices[self.toppar.nonbonded14_indices[:, 0]]],
                          x[:, self.atom_indices[self.toppar.nonbonded14_indices[:, 1]]])
        return dists

    def angles(self, x):
        a = self.angle(x[:, self.atom_indices[self.toppar.angle_indices[:, 0]]],
                       x[:, self.atom_indices[self.toppar.angle_indices[:, 1]]],
                       x[:, self.atom_indices[self.toppar.angle_indices[:, 2]]])
        return a

    def torsions(self, x):
        t = self.torsion(x[:, self.atom_indices[self.toppar.torsion_indices[:, 0]]],
                         x[:, self.atom_indices[self.toppar.torsion_indices[:, 1]]],
                         x[:, self.atom_indices[self.toppar.torsion_indices[:, 2]]],
                         x[:, self.atom_indices[self.toppar.torsion_indices[:, 3]]])
        return t

    def bondlengths_tf(self, x):
        dists = self.dist_tf(tf.gather(x, self.atom_indices[self.toppar.bond_indices[:, 0]], axis=1),
                             tf.gather(x, self.atom_indices[self.toppar.bond_indices[:, 1]], axis=1))
        return dists

    def nb_dists_tf(self, x):
        dists = self.dist_tf(tf.gather(x, self.atom_indices[self.toppar.nonbonded_indices[:, 0]], axis=1),
                             tf.gather(x, self.atom_indices[self.toppar.nonbonded_indices[:, 1]], axis=1))
        return dists

    def nb14_dists_tf(self, x):
        dists = self.dist_tf(tf.gather(x, self.atom_indices[self.toppar.nonbonded14_indices[:, 0]], axis=1),
                             tf.gather(x, self.atom_indices[self.toppar.nonbonded14_indices[:, 1]], axis=1))
        return dists

    def angles_tf(self, x):
        a = self.angle_tf(tf.gather(x, self.atom_indices[self.toppar.angle_indices[:, 0]], axis=1),
                          tf.gather(x, self.atom_indices[self.toppar.angle_indices[:, 1]], axis=1),
                          tf.gather(x, self.atom_indices[self.toppar.angle_indices[:, 2]], axis=1))
        return a

    def torsions_tf(self, x):
        t = self.torsion_tf(tf.gather(x, self.atom_indices[self.toppar.torsion_indices[:, 0]], axis=1),
                            tf.gather(x, self.atom_indices[self.toppar.torsion_indices[:, 1]], axis=1),
                            tf.gather(x, self.atom_indices[self.toppar.torsion_indices[:, 2]], axis=1),
                            tf.gather(x, self.atom_indices[self.toppar.torsion_indices[:, 3]], axis=1))
        return t

    def energy(self, x):
        E = np.zeros(x.shape[0])
        if self.toppar.bond_indices.size > 0:
            E += np.sum(self.toppar.bond_k * ((self.bondlengths(x) - self.toppar.bond_d0)**2), axis=1)
        if self.toppar.angle_indices.size > 0:
            da = (self.angles(x) - self.toppar.angle_a0) * 0.017453  # in radians
            E += np.sum(self.toppar.angle_k * (da**2), axis=1)
        if self.toppar.torsion_indices.size > 0:
            tor = np.deg2rad(self.torsions(x))
            phase = np.deg2rad(self.toppar.torsion_phase)
            E += np.sum(self.toppar.torsion_k * np.cos(self.toppar.torsion_n*tor - phase), axis=1)
        if self.toppar.nonbonded_indices.size > 0:
            dist = self.nb_dists(x)
            f6 = (self.toppar.nonbonded_sigma / dist) ** 6
            E += np.sum(self.toppar.nonbonded_epsilon * (f6**2 - 2*f6), axis=1)
        if self.toppar.nonbonded14_indices.size > 0:
            dist = self.nb14_dists(x)
            f6 = (self.toppar.nonbonded14_sigma / dist) ** 6
            E += np.sum(self.toppar.nonbonded14_epsilon / 1.2 * (f6**2 - 2*f6), axis=1)

        # alignment energy
        if self.align > 0:
            E += self.align*(x[:, 3*self.align_atoms[0]+0]**2
                             + x[:, 3*self.align_atoms[0]+1]**2
                             + x[:, 3*self.align_atoms[0]+2]**2
                             + x[:, 3*self.align_atoms[1]+1]**2
                             + x[:, 3*self.align_atoms[1]+2]**2
                             + x[:, 3*self.align_atoms[2]+2]**2)

        return E

    def energy_tf(self, x):
        batchsize = tf.shape(x)[0]
        E = tf.zeros((batchsize,))
        if self.toppar.bond_indices.size > 0:
            E += tf.reduce_sum(self.toppar.bond_k * ((self.bondlengths_tf(x) - self.toppar.bond_d0)**2), axis=1)
        if self.toppar.angle_indices.size > 0:
            da = (self.angles_tf(x) - self.toppar.angle_a0) * 0.017453  # in radians
            E += tf.reduce_sum(self.toppar.angle_k * (da**2), axis=1)
        if self.toppar.torsion_indices.size > 0:
            tor = np.float32(np.pi / 180.0) * self.torsions_tf(x)
            phase = np.float32(np.pi / 180.0) * self.toppar.torsion_phase
            E += tf.reduce_sum(self.toppar.torsion_k * tf.cos(self.toppar.torsion_n*tor - phase), axis=1)
        if self.toppar.nonbonded_indices.size > 0:
            dist = self.nb_dists_tf(x)
            f6 = (self.toppar.nonbonded_sigma / dist) ** 6
            E += tf.reduce_sum(self.toppar.nonbonded_epsilon * (f6**2 - 2*f6), axis=1)
        if self.toppar.nonbonded14_indices.size > 0:
            dist = self.nb14_dists_tf(x)
            f6 = (self.toppar.nonbonded14_sigma / dist) ** 6
            E += tf.reduce_sum(self.toppar.nonbonded14_epsilon / 1.2 * (f6**2 - 2*f6), axis=1)

        # alignment energy
        if self.align > 0:
            E += self.align*(x[:, 3*self.align_atoms[0]+0]**2
                             + x[:, 3*self.align_atoms[0]+1]**2
                             + x[:, 3*self.align_atoms[0]+2]**2
                             + x[:, 3*self.align_atoms[1]+1]**2
                             + x[:, 3*self.align_atoms[1]+2]**2
                             + x[:, 3*self.align_atoms[2]+2]**2)

        return E


class TopPar(object):

    def __init__(self, atom_names, sigma=None, epsilon=None, charge=None):
        self.atom_names = atom_names
        self.natoms = len(atom_names)
        self.bond_indices = []
        self.bond_k = []
        self.bond_d0 = []
        self.angle_indices = []
        self.angle_k = []
        self.angle_a0 = []
        self.torsion_indices = []
        self.torsion_k = []
        self.torsion_n = []
        self.torsion_phase = []
        self.sigma = sigma
        self.epsilon = epsilon
        self.charge = charge


    def finalize(self):
        # collect all bonded interactions
        self.bond_indices = np.array(self.bond_indices).astype(np.int32)
        self.bond_k = np.array(self.bond_k).astype(np.float32)
        self.bond_d0 = np.array(self.bond_d0).astype(np.float32)
        self.angle_indices = np.array(self.angle_indices).astype(np.int32)
        self.angle_k = np.array(self.angle_k).astype(np.float32)
        self.angle_a0 = np.array(self.angle_a0).astype(np.float32)
        self.torsion_indices = np.array(self.torsion_indices).astype(np.int32)
        self.torsion_k = np.array(self.torsion_k).astype(np.float32)
        self.torsion_n = np.array(self.torsion_n).astype(np.float32)
        self.torsion_phase = np.array(self.torsion_phase).astype(np.float32)
        pairs_nb, pairs_14 = self.nonbonded_pairs()
        self.nonbonded_indices = np.array(pairs_nb).astype(np.int32)
        self.nonbonded14_indices = np.array(pairs_14).astype(np.int32)
        if len(pairs_nb) > 0:
            self.nonbonded_sigma = np.zeros(len(pairs_nb), dtype=np.float32)
            self.nonbonded_epsilon = np.zeros(len(pairs_nb), dtype=np.float32)
            for i, p in enumerate(pairs_nb):
                self.nonbonded_sigma[i] = 0.5 * (self.sigma[p[0]] + self.sigma[p[1]])
                self.nonbonded_epsilon[i] = np.sqrt(self.epsilon[p[0]] * self.epsilon[p[1]])
        if len(pairs_14) > 0:
            self.nonbonded14_sigma = np.zeros(len(pairs_14), dtype=np.float32)
            self.nonbonded14_epsilon = np.zeros(len(pairs_14), dtype=np.float32)
            for i, p in enumerate(pairs_14):
                self.nonbonded14_sigma[i] = 0.5 * (self.sigma[p[0]] + self.sigma[p[1]])
                self.nonbonded14_epsilon[i] = np.sqrt(self.epsilon[p[0]] * self.epsilon[p[1]])

    def add_bond(self, indices, k, d0):
        """
        Parameters
        ----------
        indices : array
            Array of 3 ints, defining the angle
        k : float
            Force constant
        a0 : float
            Reference angle

        """
        self.bond_indices.append(indices)
        self.bond_k.append(k)
        self.bond_d0.append(d0)

    def add_angle(self, indices, k, a0):
        """
        Parameters
        ----------
        indices : array
            Array of 3 ints, defining the angle
        k : float
            Force constant
        a0 : float
            Reference angle

        """
        self.angle_indices.append(indices)
        self.angle_k.append(k)
        self.angle_a0.append(a0)

    def add_torsion(self, indices, k, n, phase):
        """
        Parameters
        ----------
        indices : array
            Array of 3 ints, defining the angle
        k : float
            Force constant
        a0 : float
            Reference angle

        """
        self.torsion_indices.append(indices)
        self.torsion_k.append(k)
        self.torsion_n.append(n)
        self.torsion_phase.append(phase)

    def nonbonded_pairs(self):
        # 0: nb, 1: 1-4, 2:bonded
        bonded = np.zeros((self.natoms, self.natoms), dtype=np.int)
        for b in self.bond_indices:
            bonded[b[0], b[1]] = 2
            bonded[b[1], b[0]] = 2
        for b in self.angle_indices:
            bonded[b[0], b[2]] = 2
            bonded[b[2], b[0]] = 2
        for b in self.torsion_indices:
            bonded[b[0], b[3]] = 1
            bonded[b[3], b[0]] = 1
        pairs_nb = []
        pairs_14 = []
        for i in range(self.natoms-1):
            for j in range(i+1, self.natoms):
                if not bonded[i,j] == 1:
                    pairs_14.append([i,j])
                if not bonded[i,j] == 0:
                    pairs_nb.append([i,j])
        return pairs_nb, pairs_14

    def mdtraj_topology(self):
        import mdtraj
        top = mdtraj.Topology()
        chain = top.add_chain()
        res = None
        element = None
        for a in self.atom_names:
            if a.startswith('C'):
                res = top.add_residue('CH2', chain)
                element = mdtraj.element.carbon
            if a.startswith('H'):
                element = mdtraj.element.hydrogen
            top.add_atom(a, element, res)
        atoms = [a for a in top.atoms]
        for i in range(self.bond_indices.shape[0]):
            b = self.bond_indices[i]
            top.add_bond(atoms[b[0]], atoms[b[1]])
        return top


# AMBER parameters from https://www.researchgate.net/profile/Alexander_Lyubartsev/publication/260253517_A_new_AMBER-compatible_force_field_parameter_set_for_alkanes/links/563319ec08ae242468da2034/A-new-AMBER-compatible-force-field-parameter-set-for-alkanes.pdf
class Amber(object):
    def __init__(self):
        #                name  sigma  eps
        self.nb_pars = [['CT', 1.75,  0.156 / 0.6],
                        ['CH3', 1.75,  0.184 / 0.6],
                        ['CH', 1.75,  0.1 / 0.6],
                        ['HC', 1.495, 0.0124 / 0.6]]
        #                   name  name  k          b0
        self.bond_pars = [['CH3', 'HC', 340 / 0.6, 1.093],
                          ['CT',  'HC', 340 / 0.6, 1.096],
                          ['CH',  'HC', 340 / 0.6, 1.097],
                          ['C*',  'C*',  240 / 0.6, 1.526]]
        #                  name  name  name  k         a0
        self.angle_pars = [['HC', 'C*', 'HC', 33 / 0.6, 107],
                           ['C*', 'C*', 'HC', 52 / 0.6, 110.7],
                           ['C*', 'C*', 'C*', 30 / 0.6, 109.5]]
        #                    name  name  name  name k        phase n
        self.torsion_pars = [['*', 'CT', 'CT', '*', 1.45 / (0.6*9.0), 0, 3],
                             ['*', 'CT', 'CH', '*', 1.45 / (0.6*9.0), 0, 3],
                             ['*', 'CH', 'CH', '*', 1.45 / (0.6*9.0), 0, 3]]

    def _name_match(self, ref, name):
        if ref == name:
            return True
        if '*' in ref:
            i = ref.index('*')
            if name.startswith(ref[:i]):
                return True
        return False

    def _bond_match(self, b, name1, name2):
        return (self._name_match(b[0], name1) and self._name_match(b[1], name2)) or \
               (self._name_match(b[1], name1) and self._name_match(b[0], name2))

    def _angle_match(self, a, name1, name2, name3):
        return (self._name_match(a[0], name1) and self._name_match(a[1], name2) and self._name_match(a[2], name3)) or \
               (self._name_match(a[2], name1) and self._name_match(a[1], name2) and self._name_match(a[0], name3))

    def _torsion_match(self, t, name1, name2, name3, name4):
        return (self._name_match(t[0], name1) and self._name_match(t[1], name2) and
                self._name_match(t[2], name3) and self._name_match(t[3], name4)) or \
               (self._name_match(t[3], name1) and self._name_match(t[2], name2) and
                self._name_match(t[1], name3) and self._name_match(t[0], name4))

    def epsilon(self, atomname):
        for par in self.nb_pars:
            if self._name_match(par[0], atomname):
                return par[2]
        raise ValueError('Unknown atom name', atomname)

    def sigma(self, atomname):
        for par in self.nb_pars:
            if self._name_match(par[0], atomname):
                return par[1]
        raise ValueError('Unknown atom name', atomname)

    def bond_k(self, atomname1, atomname2):
        for par in self.bond_pars:
            if self._bond_match(par, atomname1, atomname2):
                return par[2]
        raise ValueError('Unknown bond', atomname1, atomname2)

    def bond_b0(self, atomname1, atomname2):
        for par in self.bond_pars:
            if self._bond_match(par, atomname1, atomname2):
                return par[3]
        raise ValueError('Unknown bond', atomname1, atomname2)

    def angle_k(self, atomname1, atomname2, atomname3):
        for par in self.angle_pars:
            if self._angle_match(par, atomname1, atomname2, atomname3):
                return par[3]
        raise ValueError('Unknown angle', atomname1, atomname2, atomname3)

    def angle_a0(self, atomname1, atomname2, atomname3):
        for par in self.angle_pars:
            if self._angle_match(par, atomname1, atomname2, atomname3):
                return par[4]
        raise ValueError('Unknown angle', atomname1, atomname2, atomname3)

    def torsion_k(self, atomname1, atomname2, atomname3, atomname4):
        for par in self.torsion_pars:
            if self._torsion_match(par, atomname1, atomname2, atomname3, atomname4):
                return par[4]
        raise ValueError('Unknown torsion', atomname1, atomname2, atomname3, atomname4)

    def torsion_phase(self, atomname1, atomname2, atomname3, atomname4):
        for par in self.torsion_pars:
            if self._torsion_match(par, atomname1, atomname2, atomname3, atomname4):
                return par[5]
        raise ValueError('Unknown torsion', atomname1, atomname2, atomname3, atomname4)

    def torsion_n(self, atomname1, atomname2, atomname3, atomname4):
        for par in self.torsion_pars:
            if self._torsion_match(par, atomname1, atomname2, atomname3, atomname4):
                return par[6]
        raise ValueError('Unknown torsion', atomname1, atomname2, atomname3, atomname4)


def topology_hydrocarbon(nC, Cbonds):
    """ Builds the topology of a hydrocarbon given its backbone connectivity

    Parameters
    ----------
    nC : int
        Number of carbons
    Cbonds : list of [a1, a2]
        List of bonds between carbons

    """
    # Count bonds involving each C
    bonds_per_C = np.bincount(np.array(Cbonds).flatten())
    # Define atom types
    Cidx = np.zeros(nC, dtype=int)
    atom_names = []
    natoms = 0
    bondsCH = []
    for i in range(nC):
        Cidx[i] = natoms
        if bonds_per_C[i] == 1:
            atom_names += ['CH3']
        if bonds_per_C[i] == 2:
            atom_names += ['CT']
        if bonds_per_C[i] == 3:
            atom_names += ['CH']
        # add C-H bonds
        for j in range(4 - bonds_per_C[i]):
            atom_names += ['HC']
            bondsCH.append([natoms, natoms+1+j])
        natoms += 5 - bonds_per_C[i]
    # add C-C bonds
    bondsCC = [[Cidx[b[0]], Cidx[b[1]]] for b in Cbonds]
    bonds = bondsCC + bondsCH
    # build bond graph (adjacency list)
    bondgraph = [[] for i in range(natoms)]
    for b in bonds:
        bondgraph[b[0]].append(b[1])
        bondgraph[b[1]].append(b[0])
    # list all angles
    angles = []
    for i2 in range(len(bondgraph)):
        for i1 in bondgraph[i2]:
            for i3 in bondgraph[i2]:
                if i1 != i3 and i1 < i3:
                    angles.append([i1, i2, i3])
    # list all torsions
    torsions = []
    for b in bonds:
        i2 = b[0]
        i3 = b[1]
        for i1 in bondgraph[i2]:
            for i4 in bondgraph[i3]:
                if i1 != i3 and i2 != i4 and i1 != i4:
                    torsions.append([i1, i2, i3, i4])
    return atom_names, bonds, angles, torsions


def build_hydrocarbon(nC, Cbonds):
    atom_names, bonds, angles, torsions = topology_hydrocarbon(nC, Cbonds)
    # Parameter set
    amber = Amber()
    # make toppar
    sigma = [amber.sigma(aname) for aname in atom_names]
    epsilon = [amber.epsilon(aname) for aname in atom_names]
    toppar = TopPar(atom_names, sigma=sigma, epsilon=epsilon)
    for b in bonds:
        toppar.add_bond([b[0], b[1]],
                        amber.bond_k(atom_names[b[0]], atom_names[b[1]]),
                        amber.bond_b0(atom_names[b[0]], atom_names[b[1]]))
    for a in angles:
        toppar.add_angle([a[0], a[1], a[2]],
                         amber.angle_k(atom_names[a[0]], atom_names[a[1]], atom_names[a[2]]),
                         amber.angle_a0(atom_names[a[0]], atom_names[a[1]], atom_names[a[2]]))
    for d in torsions:
        toppar.add_torsion([d[0], d[1], d[2], d[3]],
                           amber.torsion_k(atom_names[d[0]], atom_names[d[1]], atom_names[d[2]], atom_names[d[3]]),
                           amber.torsion_n(atom_names[d[0]], atom_names[d[1]], atom_names[d[2]], atom_names[d[3]]),
                           amber.torsion_phase(atom_names[d[0]], atom_names[d[1]], atom_names[d[2]], atom_names[d[3]]))
    toppar.finalize()
    return toppar