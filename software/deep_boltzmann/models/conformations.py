import numpy as np

class RotamerMapper(object):
    def __init__(self, ntorsion, rotamers_per_torsion=3, cyclic=True, mirror=True):
        self.ntorsion = ntorsion
        self.rotamers_per_torsion = rotamers_per_torsion
        self.cyclic = cyclic
        self.all_keys = self._all_possible_rotamers()
        if self.cyclic:
            self.map2unique = np.concatenate([self._cyclic_map(self.all_keys), np.array([-1])]).astype(int)
        else:
            self.map2unique = np.concatenate([np.arange(len(self.all_keys)), np.array([-1])]).astype(int)
        
    def _rot2index(self, rotamer):
        """ Converts the rotamer key to the index it encodes
        
        Parameters
        ----------
        rotamer : array of int
            Array encoding setting of each rotamer, e.g. [0, 0, 2, 1, 1]
        
        Returns
        -------
        index : int
            Directly encoded key, e.g. with rotamers_per_torsion=3 the rotamer is interpreted as a trinary number,
            00211 -> 2*9 + 1*3 + 1*1 = 22.
            
        """
        if np.ndim(rotamer) == 1:
            rotamer = np.array([rotamer])
        v = np.zeros(rotamer.shape[0])
        for p in range(self.ntorsion):
            v += rotamer[:, p] * (self.rotamers_per_torsion**(self.ntorsion-1-p))
        # sets -1 where -1 rotamers are found
        v = np.where(np.min(rotamer, axis=1) < 0, -1, v).astype(int)
        
        return v
    
    def _all_possible_rotamers(self):
        """ Lists all possible rotamers
        """
        keys = [np.zeros(self.ntorsion, dtype=int)]
        for i in range(1, self.rotamers_per_torsion**self.ntorsion):
            key = keys[-1].copy()
            for p in range(self.ntorsion-1, -1, -1):
                if key[p] < self.rotamers_per_torsion-1:
                    key[p] += 1
                    break
                else:
                    key[p] = 0
            keys.append(key)
        return keys
    
    def _cyclic_map(self, all_keys, mirror=True):
        """ Creates a uniqueness map for cyclic and mirror permutation """
        map2unique = np.zeros(len(all_keys), dtype=int)
        for i in range(len(all_keys)):
            # list all equivalent keys
            eqkeys = [np.roll(all_keys[i], j) for j in range(self.ntorsion)]
            if mirror:
                keyinv = all_keys[i][::-1]
                eqkeys += [np.roll(keyinv, j) for j in range(self.ntorsion)]
            # choose target
            eqidx = self._rot2index(np.vstack(eqkeys))
            map2unique[i] = int(eqidx.min())
        return map2unique

    def distinct_rotamers(self):
        return len(set(list(self.map2unique)))
    
    def torsion2rotamer(self, torsion, width=100):
        """ Maps torsion or array of torsions to rotamers """
        if self.rotamers_per_torsion != 3:
            raise NotImplementedError('Not implemented number of rotamers per torsion')
        bins = np.array([-180+0.5*width, -60-0.5*width, -60+0.5*width, 60-0.5*width, 60+0.5*width, 180-0.5*width])
        assignment1 = np.digitize(torsion, bins)
        indices = np.array([0, -1, 1, -1, 2, -1, 0])
        assignment2 = indices[assignment1]
        return assignment2 
    
    def rotamer2index(self, rotamer):
        return self.map2unique[self._rot2index(rotamer)]
        
    def torsion2index(self, torsion, width=100):
        rot = self.torsion2rotamer(torsion, width=width)
        return self.rotamer2index(rot)

    def histogram(self, indices):
        distinct_indices = np.array(list(set(list(indices))))
        distinct_indices.sort()
        if distinct_indices[0] == -1:
            distinct_indices = distinct_indices[1:]
        counts = np.zeros(distinct_indices.size, dtype=int)
        mapped_indices = np.searchsorted(distinct_indices, indices)
        for i in range(len(mapped_indices)):
            m = mapped_indices[i]
            if indices[i] != -1:
                counts[m] += 1
        keys = [self.all_keys[i] for i in distinct_indices]
        return distinct_indices, keys, counts