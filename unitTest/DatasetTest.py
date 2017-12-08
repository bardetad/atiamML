import unittest
import sys
# Add the src folder path to the sys.path list
sys.path.append('../src/dataset')


from ManageDataset import NPZ_Dataset


class TestNPZDataset(unittest.TestCase):

    def testDatasetLength(self):
        dataset = NPZ_Dataset('dummyDataset_100.npz',
                              './dummyDataset/', 'Spectrums', 'labels')
        self.assertTrue(len(dataset) == 100)

    def testDatasetLoad(self):
        dataset = NPZ_Dataset('dummyDataset_100.npz',
                              './dummyDataset/', 'Spectrums', 'labels')
        for i in range(5):
            data = dataset[i]
            print(i, data['image'], data['label'])
        self.assertTrue(dataset != [])


suiteNPZDataset = unittest.TestLoader().loadTestsFromTestCase(TestNPZDataset)
print "\n\n------------------- Dataset from .npz file Test Suite -------------------\n"
unittest.TextTestRunner(verbosity=2).run(suiteNPZDataset)
