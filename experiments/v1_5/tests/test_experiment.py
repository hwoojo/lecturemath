import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np
from experiments.v1_5 import methods
from experiments.v1_5.adapter import load_mask, validate_probability, LectureNetAdapter
from experiments.v1_5.cli import parser, run
from experiments.v1_5.evaluation import evaluate


class MethodsTest(unittest.TestCase):
    def test_shapes_constants_small_and_determinism(self):
        for shape in [(1,1),(2,3),(32,48)]:
            for value in [0,127,255]:
                image = np.full((*shape,3),value,np.uint8)
                for polarity in ['dark','light']:
                    for function in [methods.otsu,methods.sauvola]:
                        a = function(image,polarity)
                        self.assertEqual(a.shape,shape)
                        np.testing.assert_array_equal(a,function(image,polarity))
                        self.assertTrue(set(np.unique(methods.png(a))) <= {0,255})

    def test_polarity_and_hybrid_guarantees(self):
        image = np.full((64,80,3),255,np.uint8)
        image[20:40,30:33] = 0
        for fn in [methods.otsu,methods.sauvola]:
            dark = fn(image,'dark')
            np.testing.assert_array_equal(dark,fn(255-image,'light'))
            self.assertTrue(dark[25,31])
            self.assertFalse(dark[5,5])
        base = np.zeros(image.shape[:2],bool)
        base[22:38,31] = True
        base[3,3] = True  # preserve even an incorrect original pixel
        out = methods.hybrid(image,base,radius=2)
        self.assertTrue(np.all(out[base]))
        region = cv2.dilate(base.astype(np.uint8),np.ones((5,5),np.uint8)) > 0
        self.assertFalse((out & ~region).any())
        self.assertGreater(out.sum(),base.sum())
        np.testing.assert_array_equal(methods.hybrid(image,base,radius=0),base)

    def test_evaluation(self):
        pred = np.array([[1,1],[0,0]],bool)
        truth = np.array([[1,0],[1,0]],bool)
        roi = np.ones((2,2),bool)
        result = evaluate(pred,roi,truth)
        self.assertEqual(result['f1'],.5)
        self.assertAlmostEqual(result['iou'],1/3)
        self.assertNotIn('f1',evaluate(pred,roi))
        roi[1] = False
        self.assertEqual(evaluate(pred,roi,truth)['recall'],1)
        with self.assertRaises(ValueError):
            evaluate(pred,roi,np.zeros((3,3),bool))
        with self.assertRaises(ValueError):
            evaluate(pred,np.zeros((2,2),bool))

    def test_adapter_rejects_invalid_data(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'mask.png'
            cv2.imwrite(str(path),np.array([[0,255]],np.uint8))
            np.testing.assert_array_equal(load_mask(path,(1,2),'dark'),[[True,False]])
            np.testing.assert_array_equal(load_mask(path,(1,2),'light'),[[False,True]])
            with self.assertRaises(ValueError):
                load_mask(path,(2,1),'dark')
            cv2.imwrite(str(path),np.array([[127,255]],np.uint8))
            with self.assertRaises(ValueError):
                load_mask(path,(1,2),'dark')
        for invalid in [np.zeros((2,2)),np.full((2,2),np.nan),np.full((2,2),2)]:
            with self.assertRaises(ValueError):
                validate_probability(invalid,(2,2))

    def test_cli_without_checkpoint_and_with_fixture_baseline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = np.full((64,96,3),255,np.uint8)
            cv2.putText(image,'x=1',(5,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1)
            cv2.imwrite(str(root/'frame.png'),image)
            args = parser().parse_args([str(root/'frame.png'),'--output',str(root/'out'),'--polarity','dark'])
            with contextlib.redirect_stdout(io.StringIO()):
                result = run(args)
            self.assertEqual({r['method'] for r in result['skipped']},{'B0','B3','B4'})
            self.assertTrue(all('f1' not in m for m in result['metrics']))
            (root/'base').mkdir()
            (root/'prob').mkdir()
            base = methods.otsu(image)
            cv2.imwrite(str(root/'base/frame.png'),methods.png(base))
            np.save(root/'prob/frame.npy',np.where(base,.8,.1).astype(np.float32))
            args.baseline_dir = root/'base'
            args.probability_dir = root/'prob'
            args.roi = (0,0,80,60)
            with contextlib.redirect_stdout(io.StringIO()):
                result = run(args)
            self.assertFalse(result['skipped'])
            for record in result['metrics']:
                p = cv2.imread(str(root/'out/frame'/(record['method']+'.png')),0)
                self.assertEqual(p.shape,image.shape[:2])
                if record['method'].startswith('B3'):
                    self.assertTrue(np.all(p[base] == 0))
            self.assertTrue((root/'out/frame/contact_sheet.png').exists())
            args.gt_dir = root/'base'
            with contextlib.redirect_stdout(io.StringIO()):
                result = run(args)
            self.assertEqual(next(r for r in result['metrics'] if r['method']=='B0')['f1'],1.)
            args.resolution = '1080p'
            with self.assertRaises(ValueError):
                run(args)

    def test_postprocessing_does_not_import_torch(self):
        subprocess.run([sys.executable,'-c',"from experiments.v1_5 import cli; import sys; assert 'torch' not in sys.modules"],check=True)


class V1RegressionTest(unittest.TestCase):
    def test_adapter_matches_unchanged_v1_and_cache(self):
        # Architecture fixture with small channel counts, random weights: inference ONLY.
        try:
            import torch
        except ImportError:
            self.skipTest('Torch unavailable; independent experiments still supported')
        sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'ACCESS2021_release'))
        from AM_CommonTools.configuration.configuration import Configuration
        from AccessMath.lecturenet_v1.FCN_lecturenet import FCN_LectureNet
        from PIL import Image
        torch.set_num_threads(2)
        torch.manual_seed(17)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root/'fixture.conf'
            original = Path('ACCESS2021_release/configs/FCN_LectureNet.conf').read_text()
            lines = []
            for line in original.splitlines():
                if line.startswith('FCN_BINARIZER_NET_') and ('FILTERS' in line or 'PIXEL_FEATURES' in line):
                    line = line.split('=')[0]+'= 2'
                lines.append(line)
            config.write_text('\n'.join(lines))
            model = FCN_LectureNet.CreateFromConfig(Configuration.from_file(str(config)),3,False).eval()
            checkpoint = root/'fixture.dat'
            torch.save(model.state_dict(),checkpoint)
            adapter = LectureNetAdapter(config,checkpoint)
            image = np.random.default_rng(0).integers(0,256,(64,96,3),dtype=np.uint8)
            expected = model.binarize(Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)),force_binary=True)
            actual, probability = adapter.infer(image)
            np.testing.assert_array_equal(actual,expected == 0)
            self.assertEqual(probability.shape,image.shape[:2])
            with torch.inference_mode():
                logits, _, _ = model(model.prepare_image(Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))))
            np.testing.assert_allclose(probability,1-torch.sigmoid(logits)[0,0].numpy())
            from AccessMath.preprocessing.video_worker.FCN_lecturenet_binarizer import FCN_LectureNet_Binarizer
            worker = FCN_LectureNet_Binarizer(model)
            worker.initialize(image.shape[1],image.shape[0])
            worker.handleFrame(image,None,0,0,0,0)
            np.testing.assert_array_equal(worker.last_binary,255-expected)
            cv2.imwrite(str(root/'input.png'),image)
            args = parser().parse_args([str(root/'input.png'),'--output',str(root/'out'),'--polarity','dark',
                '--methods','B0','B4','--checkpoint',str(checkpoint),'--config',str(config)])
            with contextlib.redirect_stdout(io.StringIO()):
                first = run(args)
                second = run(args)
            self.assertFalse(first['sources'][0]['inference']['cache_hit'])
            self.assertTrue(second['sources'][0]['inference']['cache_hit'])
            np.testing.assert_array_equal(cv2.imread(str(root/'out/input/B0.png'),0),methods.png(actual))


if __name__ == '__main__':
    unittest.main()
