from cv2 import data
import tensorflow as tf
import cv2, os, random, math, numbers
import numpy as np
from pathlib import Path

interp_list = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]

def maybe_random_interp(cv2_interp):
    if True:
        return np.random.choice(interp_list)
    else:
        return cv2_interp


class RandomAffine(object):
    """
    Random affine translation
    """
    def __init__(self, degrees, translate=None, scale=None, shear=None, flip=None, resample=False, fillcolor=0, with_bg=False):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor
        self.flip = flip
        self.with_bg = with_bg

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, flip, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = (random.uniform(scale_ranges[0], scale_ranges[1]),
                     random.uniform(scale_ranges[0], scale_ranges[1]))
        else:
            scale = (1.0, 1.0)

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        if flip is not None:
            flip = (np.random.rand(2) < flip).astype(np.int) * 2 - 1

        return angle, translations, scale, shear, flip

    def __call__(self, sample):
        fg, bg, alpha = sample['fg'], sample['bg'], sample['alpha']
        rows, cols, ch = fg.shape
        if np.maximum(rows, cols) < 1024:
            params = self.get_params((0, 0), self.translate, self.scale, self.shear, self.flip, fg.size)
        else:
            params = self.get_params(self.degrees, self.translate, self.scale, self.shear, self.flip, fg.size)

        center = (cols * 0.5 + 0.5, rows * 0.5 + 0.5)
        M = self._get_inverse_affine_matrix(center, *params)
        M = np.array(M).reshape((2, 3))

        fg = cv2.warpAffine(fg, M, (cols, rows),
                            flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)
        alpha = cv2.warpAffine(alpha, M, (cols, rows),
                               flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)

        if(self.with_bg):
            bg = cv2.warpAffine(bg, M, (cols, rows),
                    flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)

        sample['fg'], sample['bg'], sample['alpha'] = fg, bg, alpha

        return sample


    @ staticmethod
    def _get_inverse_affine_matrix(center, angle, translate, scale, shear, flip):
        # Helper method to compute inverse matrix for affine transformation

        # As it is explained in PIL.Image.rotate
        # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
        # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
        # C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
        # RSS is rotation with scale and shear matrix
        # It is different from the original function in torchvision
        # The order are changed to flip -> scale -> rotation -> shear
        # x and y have different scale factors
        # RSS(shear, a, scale, f) = [ cos(a + shear)*scale_x*f -sin(a + shear)*scale_y     0]
        # [ sin(a)*scale_x*f          cos(a)*scale_y             0]
        # [     0                       0                      1]
        # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

        angle = math.radians(angle)
        shear = math.radians(shear)
        scale_x = 1.0 / scale[0] * flip[0]
        scale_y = 1.0 / scale[1] * flip[1]

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
        matrix = [
            math.cos(angle) * scale_x, math.sin(angle + shear) * scale_x, 0,
            -math.sin(angle) * scale_y, math.cos(angle + shear) * scale_y, 0
        ]
        matrix = [m / d for m in matrix]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]

        return matrix

class RandomJitter(object):
    """
    Random change the hue of the image
    """

    def __init__(self, with_bg=False) -> None:
        self.with_bg = with_bg

    def __call__(self, sample):
        fg, bg, alpha = sample['fg'], sample['bg'], sample['alpha']
        # if alpha is all 0 skip
        if np.all(alpha==0):
            return sample
        # convert to HSV space, convert to float32 image to keep precision during space conversion.
        fg = cv2.cvtColor(fg.astype(np.float32)/255.0, cv2.COLOR_BGR2HSV)
        if(self.with_bg):
            bg = cv2.cvtColor(bg.astype(np.float32)/255.0, cv2.COLOR_BGR2HSV)
        # Hue noise
        hue_jitter = np.random.randint(-40, 40)
        fg[:, :, 0] = np.remainder(fg[:, :, 0].astype(np.float32) + hue_jitter, 360)
        if(self.with_bg):
            bg[:, :, 0] = np.remainder(bg[:, :, 0].astype(np.float32) + hue_jitter, 360)
        # Saturation noise
        sat_bar = fg[:, :, 1][alpha > 0].mean()
        sat_jitter = np.random.rand()*(1.1 - sat_bar)/5 - (1.1 - sat_bar) / 10
        sat = fg[:, :, 1]
        sat = np.abs(sat + sat_jitter)
        sat[sat>1] = 2 - sat[sat>1]
        fg[:, :, 1] = sat
        if(self.with_bg):
            sat = bg[:, :, 1]
            sat = np.abs(sat + sat_jitter)
            sat[sat>1] = 2 - sat[sat>1]
            bg[:, :, 1] = sat
        # Value noise
        val_bar = fg[:, :, 2][alpha > 0].mean()
        val_jitter = np.random.rand()*(1.1 - val_bar)/5-(1.1 - val_bar) / 10
        val = fg[:, :, 2]
        val = np.abs(val + val_jitter)
        val[val>1] = 2 - val[val>1]
        fg[:, :, 2] = val
        # convert back to BGR space
        fg = cv2.cvtColor(fg, cv2.COLOR_HSV2BGR)
        sample['fg'] = fg*255
        if(self.with_bg):
            val = bg[:, :, 2]
            val = np.abs(val + val_jitter)
            val[val>1] = 2 - val[val>1]
            bg[:, :, 2] = val
            # convert back to BGR space
            bg = cv2.cvtColor(bg, cv2.COLOR_HSV2BGR)
            sample['bg'] = bg*255
        return sample


class RandomHorizontalFlip(object):
    """
    Random flip image and label horizontally
    """
    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self, sample):
        fg, alpha = sample['fg'], sample['alpha']
        if np.random.uniform(0, 1) < self.prob:
            fg = cv2.flip(fg, 1)
            alpha = cv2.flip(alpha, 1)
        sample['fg'], sample['alpha'] = fg, alpha

        return sample


class CropPad512():
    def __init__(self, prob=0.5, ratio=0.2, with_bg=False) -> None:
        self.prob = prob
        self.ratio = ratio
        self.with_bg = with_bg

    def __call__(self, sample):
        fg, bg, alpha = sample['fg'], sample['bg'], sample['alpha']
        if(len(fg.shape)==2):
            h, w = fg.shape
        else:
            h, w, _ = fg.shape

        if(h<=w):
            if(random.randint(0, 1)<self.prob): #crop
                border = w - random.randint(0, int(w*self.ratio))
                if(h>border):
                    ry = random.randint(0, h-border)
                    rx = random.randint(0, w-border)
                    fg = fg[ry:ry+border, rx:rx+border, ...]
                    alpha = alpha[ry:ry+border, rx:rx+border, ...]
                    if(self.with_bg):
                        bg = bg[ry:ry+border, rx:rx+border, ...]
                else:
                    padding = random.randint(0, border-h)
                    rx = random.randint(0, w-border)
                    fg = np.pad(fg, ((padding, border-padding-h), (0, 0), (0, 0)))
                    alpha = np.pad(alpha, ((padding, border-padding-h), (0, 0)))
                    fg = fg[:, rx:rx+border, ...]
                    alpha = alpha[:, rx:rx+border, ...]
                    if(self.with_bg):
                        bg = np.pad(bg, ((padding, border-padding-h), (0, 0), (0, 0)))
                        bg = bg[:, rx:rx+border, ...]
            else:   #padding
                border = w + random.randint(0, int(w*self.ratio))
                ry = random.randint(0, border-h)
                rx = random.randint(0, border-w)
                fg = np.pad(fg, ((ry, border-ry-h), (rx, border-rx-w), (0, 0)))
                alpha = np.pad(alpha, ((ry, border-ry-h), (rx, border-rx-w)))
                if(self.with_bg):
                    bg = np.pad(bg, ((ry, border-ry-h), (rx, border-rx-w), (0, 0)))
        else:
            if(random.randint(0, 1)<self.prob): #crop
                border = h - random.randint(0, int(h*self.ratio))
                if(w>border):
                    ry = random.randint(0, h-border)
                    rx = random.randint(0, w-border)
                    fg = fg[ry:ry+border, rx:rx+border, ...]
                    alpha = alpha[ry:ry+border, rx:rx+border, ...]
                    if(self.with_bg):
                        bg = bg[ry:ry+border, rx:rx+border, ...]
                else:
                    padding = random.randint(0, border-w)
                    ry = random.randint(0, h-border)
                    fg = np.pad(fg, ((0, 0), (padding, border-padding-w), (0, 0)))
                    alpha = np.pad(alpha, ((0, 0), (padding, border-padding-w)))
                    fg = fg[ry:ry+border, :, ...]
                    alpha = alpha[ry:ry+border, :, ...]
                    if(self.with_bg):
                        bg = np.pad(bg, ((0, 0), (padding, border-padding-w), (0, 0)))
                        bg = bg[ry:ry+border, :, ...]
            else:   #padding
                border = h + random.randint(0, int(h*self.ratio))
                ry = random.randint(0, border-w)
                rx = random.randint(0, border-h)
                fg = np.pad(fg, ((rx, border-rx-h), (ry, border-ry-w), (0, 0)))
                alpha = np.pad(alpha, ((rx, border-rx-h), (ry, border-ry-w)))
                if(self.with_bg):
                    bg = np.pad(bg, ((rx, border-rx-h), (ry, border-ry-w), (0, 0)))

        sample['fg'], sample['bg'], sample['alpha'] = cv2.resize(fg, (512, 512)), cv2.resize(bg, (512, 512)), cv2.resize(alpha, (512, 512))
        return sample


class Composite():
    def __call__(self, sample):
        fg, bg, alpha = sample['fg'], sample['bg'], sample['alpha']
        alpha[alpha < 0 ] = 0
        alpha[alpha > 1] = 1
        fg[fg < 0 ] = 0
        fg[fg > 255] = 255
        bg[bg < 0 ] = 0
        bg[bg > 255] = 255

        ry = random.randint(0, bg.shape[0]-fg.shape[0])
        rx = random.randint(0, bg.shape[1]-fg.shape[1])
        bg = bg[ry:ry+fg.shape[0], rx:rx+fg.shape[0], ...]

        image = fg * alpha[:, :, None] + bg * (1 - alpha[:, :, None])
        sample['image'] = image.astype(np.uint8)
        sample['fg'] = fg.astype(np.uint8)
        sample['bg'] = bg.astype(np.uint8)
        sample['alpha'] = alpha.astype(np.float32)
        return sample


class DataGenerator():
    def __init__(self) -> None:
        self.randomAffine = RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5)
        self.randomJitter = RandomJitter()
        self.cropPad512 = CropPad512()
        self.composite = Composite()

        self.randomAffineAISeg = RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5, with_bg=True)
        self.randomJitterAISeg = RandomJitter(with_bg=True)
        self.cropPad512AISeg = CropPad512(with_bg=True)

        root_path = Path(r'E:\CVDataset\RealWorldPortrait-636')
        self.alphas = list((root_path / 'alpha').glob('*'))
        self.fgs = list((root_path / 'image').glob('*'))
        self.bgs = list(Path(r'E:\CVDataset\Backgrounds').glob('*'))

        self.alphas.sort()
        self.fgs.sort()

        root_path = Path(r'E:\CVDataset\matting_human_half')
        self.fgs2 = list((root_path / 'matting').rglob('*.png'))
        self.bgs2 = list((root_path / 'clip_img').rglob('*.jpg'))

        self.fgs2.sort()
        self.bgs2.sort()

    def iterator(self):

        for i, alpha in enumerate(self.alphas):
            fg = cv2.imread(str(self.fgs[i]))
            alpha = cv2.imread(str(alpha))[..., 0]/255.0
            bg = cv2.imread(str(self.bgs[random.randint(0, len(self.bgs)-1)]))
            sample = {'fg': fg, 'alpha': alpha, 'bg': bg, 'image_name': self.fgs[i].name}
            sample = self.randomAffine(sample)
            sample = self.randomJitter(sample)
            sample = self.cropPad512(sample)
            sample = self.composite(sample)

            yield sample['image'], sample['fg'], sample['bg'], sample['alpha'], sample['image_name']

        for i, fg in enumerate(self.fgs2):
            fg = cv2.imread(str(self.fgs2[i]), cv2.IMREAD_UNCHANGED)
            bg = cv2.imread(str(self.bgs2[i]))
            alpha = fg[..., -1]/255.0
            fg = (fg[..., :3] * (fg[..., 3:]/255.0)).astype(np.uint8)
            sample = {'fg': fg, 'alpha': alpha, 'bg': bg, 'image_name': self.fgs2[i].name}
            sample = self.randomAffineAISeg(sample)
            sample = self.randomJitterAISeg(sample)
            sample = self.cropPad512AISeg(sample)
            sample = self.composite(sample)

            yield sample['image'], sample['fg'], sample['bg'], sample['alpha'], sample['image_name']

    def prepare_dataset(self):
        dataset = tf.data.Dataset.from_generator(self.iterator, 
                                                output_signature=(tf.TensorSpec([512, 512, 3], dtype=tf.uint8),
                                                                  tf.TensorSpec([512, 512, 3], dtype=tf.uint8),
                                                                  tf.TensorSpec([512, 512, 3], dtype=tf.uint8),
                                                                  tf.TensorSpec([512, 512], dtype=tf.float32),
                                                                  tf.TensorSpec([], dtype=tf.string)))
        dataset = dataset.shuffle(64).repeat(-1).batch(6)
        self.data_iter = iter(dataset)

    def next_batch(self):
        return next(self.data_iter)


if(__name__=='__main__'):
    generator = DataGenerator()
    generator.prepare_dataset()
    for i in range(1):
        image, fg, bg, alpha, image_name = generator.next_batch()
        # print(i, image_name, image.shape)
        cv2.imshow('', image.numpy()[1, ...])
        cv2.waitKey(0)
        cv2.imshow('', tf.cast(alpha*255, tf.uint8).numpy()[1, ...])
        cv2.waitKey(0)
        cv2.imshow('', fg.numpy()[1, ...])
        cv2.waitKey(0)
        cv2.imshow('', bg.numpy()[1, ...])
        cv2.waitKey(0)
