import unittest
import run_experiment as exp
class Tests(unittest.TestCase):
 def test_deterministic(self): self.assertEqual(exp.make_split(4,20,.1),exp.make_split(4,20,.1))
 def test_size(self): self.assertEqual(len(exp.make_split(2,23,.2)),23)
 def test_correction_bounded(self): self.assertTrue(-.25<=exp.correction_from(exp.make_split(1,100,.1))<=.25)
 def test_metrics_bounded(self):
  rows=exp.make_split(5,100,.1); m=exp.metrics(rows,.02,True)
  for k in ('accuracy','brier','ece_10bin'): self.assertTrue(0<=m[k]<=1,(k,m[k]))
if __name__=='__main__': unittest.main()
