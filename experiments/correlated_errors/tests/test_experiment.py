import unittest
import run_experiment as exp
class Tests(unittest.TestCase):
 def test_deterministic(self): self.assertEqual(exp.make_rows(3,.5,20),exp.make_rows(3,.5,20))
 def test_size(self): self.assertEqual(len(exp.make_rows(1,0,17)),17)
 def test_metrics_bounds(self):
  rows=exp.make_rows(2,.25,100)
  for policy in ('majority','raw_weighted','calibrated_weighted'):
   m=exp.score(rows,policy,.03)
   for k in ('accuracy','brier','ece_10bin'): self.assertTrue(0<=m[k]<=1,(policy,k,m[k]))
if __name__=='__main__': unittest.main()
