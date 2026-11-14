"""Encode every image once with the frozen DermLIP ViT-B/16 encoder and cache the features.

    python encode_features.py --data-dir path/to/datasets

Expects, anywhere under --data-dir:
    fitzpatrick17k.zip and fitzpatrick17k.csv   (Fitzpatrick17k)
    ddi_metadata.csv and the DDI images         (DDI)
    meta.csv and the Derm7pt image folders      (Derm7pt)

Writes the caches to caches/. An existing cache is reused unless --force is given.
"""
import argparse
import glob
import io
import json
import os
import shutil
import time
import zipfile

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--data-dir', default=os.path.join(REPO, 'datasets'),
                help='directory holding the raw datasets (searched recursively)')
ap.add_argument('--out-dir', default=os.path.join(REPO, 'caches'),
                help='where the feature caches are written')
ap.add_argument('--force', action='store_true',
                help='re-encode even if the caches already exist')
ap.add_argument('--batch-size', type=int, default=32)
args = ap.parse_args()

DATA_DIR = os.path.abspath(args.data_dir)
OUT_DIR = os.path.join(os.path.abspath(args.out_dir), 'fairness_cache')
D7_DIR = os.path.join(os.path.abspath(args.out_dir), 'derm7pt_dermlip_cache')
for d in (OUT_DIR, D7_DIR):
    os.makedirs(d, exist_ok=True)

# Keep the default torch thread count: it sets the floating-point reduction order, and
# the monotonicity metric is sensitive to it.
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.pop(v, None)

import torch  # noqa: E402
import open_clip      # noqa: E402
from PIL import Image  # noqa: E402

MODEL_NAME = 'hf-hub:redlessone/DermLIP_ViT-B-16'
BATCH_SIZE = args.batch_size
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEV_TYPE = DEVICE
zf = None

print('datasets: %s' % DATA_DIR)
print('caches:   %s' % os.path.dirname(OUT_DIR))
print('torch %s, %d threads, device %s'
      % (torch.__version__, torch.get_num_threads(), DEVICE))


def find_one(name, roots=None):
    """First file called `name` anywhere under the dataset directory, or None."""
    for r in (roots or [DATA_DIR]):
        hits = sorted(glob.glob(os.path.join(r, '**', name), recursive=True))
        if hits:
            return hits[0]
    return None


_need = ['features_fitz.npz', 'features_ddi.npz', 'features_derm7pt.npz',
         'meta_fitz.parquet', 'meta_ddi.parquet']
_have = {n: os.path.join(OUT_DIR, n) for n in _need
         if os.path.exists(os.path.join(OUT_DIR, n))}
ENCODE = args.force or len(_have) < len(_need)

if not ENCODE:
    print('caches complete; nothing to encode. Pass --force to re-encode.')
else:
    missing = [n for n in _need if n not in _have]
    if missing:
        print('encoding (%d cache file(s) absent: %s)' % (len(missing), ', '.join(missing)))

if ENCODE and DEVICE == 'cuda':
    cap = torch.cuda.get_device_capability()
    sm = 'sm_%d%d' % cap
    archs = torch.cuda.get_arch_list()
    print('GPU: %s (compute %d.%d -> %s)'
          % (torch.cuda.get_device_name(0), cap[0], cap[1], sm))
    if sm not in archs:
        print('WARNING: %s is not in this torch build (%s). Falling back to CPU, which '
              'takes hours rather than minutes; install a torch built for %s instead.'
              % (sm, ' '.join(archs), sm))
        DEVICE, DEV_TYPE = 'cpu', 'cpu'
elif ENCODE:
    print('no GPU detected: encoding on CPU takes hours rather than minutes.')

model = tokenizer = preprocess = None
if ENCODE:
    print('loading DermLIP ...')
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model = model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False
    print('logit_scale.exp() = %.2f' % float(model.logit_scale.exp()))

# --- prompt scores (prompts in src/prompts.py) and image encoding ---
from src.prompts import CONCEPT_PROMPTS
MALIG_PROMPTS = ('a photo of a malignant skin lesion', 'a photo of a benign skin lesion')


def _open(src):
    """Open an image from a file path or a (ZipFile, member name) pair."""
    if isinstance(src, tuple):
        zf, member = src
        return Image.open(io.BytesIO(zf.read(member))).convert('RGB')
    return Image.open(src).convert('RGB')


def encode_images(srcs):
    """Return (embeddings, mask); a failed load keeps a zero row with mask False."""
    n = len(srcs); feats, order, i = [], [], 0
    while i < n:
        tens, idxs = [], []
        for j, s in enumerate(srcs[i:i + BATCH_SIZE]):
            if not s:
                continue
            try:
                tens.append(preprocess(_open(s))); idxs.append(i + j)
            except Exception:
                pass
        if tens:
            x = torch.stack(tens).to(DEVICE)
            with torch.no_grad(), torch.autocast(device_type=DEV_TYPE,
                                                 enabled=(DEV_TYPE == 'cuda')):
                f = model.encode_image(x)
                f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.float().cpu().numpy()); order.extend(idxs)
        i += BATCH_SIZE
        if (i // BATCH_SIZE) % 50 == 0:
            print('   ', min(i, n), '/', n, flush=True)
    feats = np.concatenate(feats, 0) if feats else np.zeros((0, 512), np.float32)
    d = feats.shape[1] if feats.shape[0] else 512
    emb = np.zeros((n, d), np.float32); mask = np.zeros(n, bool)
    for k, idx in enumerate(order):
        emb[idx] = feats[k]; mask[idx] = True
    return emb, mask


def encode_text(prompts):
    tok = tokenizer(list(prompts)).to(DEVICE)
    with torch.no_grad(), torch.autocast(device_type=DEV_TYPE, enabled=(DEV_TYPE == 'cuda')):
        t = model.encode_text(tok)
        t = t / t.norm(dim=-1, keepdim=True)
    return t.float().cpu().numpy()


def softmax2(a, b):
    z = np.stack([a, b], 1) * 100.0
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def concept_scores_zs(emb, mask):
    names = list(CONCEPT_PROMPTS)
    pres = encode_text([CONCEPT_PROMPTS[k][0] for k in names])
    absn = encode_text([CONCEPT_PROMPTS[k][1] for k in names])
    out = np.zeros((emb.shape[0], len(names)), np.float32)
    for c in range(len(names)):
        out[:, c] = softmax2(absn[c] @ emb.T, pres[c] @ emb.T)[:, 1]
    out[~mask] = np.nan
    return out, names


def malig_prob(emb, mask):
    t = encode_text(list(MALIG_PROMPTS))
    out = softmax2(t[1] @ emb.T, t[0] @ emb.T)[:, 1]
    out[~mask] = np.nan
    return out


def cache_dataset(tag, image_id, srcs, malignant, group, split, extra=None):
    """Encode one dataset and write its cache, keyed by image_id."""
    assert len(image_id) == len(srcs), 'image_id must be one per image'
    print('[%s] encoding %d images ...' % (tag, len(srcs)))
    t0 = time.time()
    emb, mask = encode_images(srcs)
    cs, _ = concept_scores_zs(emb, mask)
    mp = malig_prob(emb, mask)
    print('  ok: %d / %d in %.1fs' % (int(mask.sum()), len(srcs), time.time() - t0))
    payload = dict(emb=emb, mask=mask, concept_scores=cs,
                   image_id=np.asarray([str(v) for v in image_id], dtype=object),
                   malignant=np.asarray(malignant, np.int64),
                   group=np.asarray(group, 'U8'), split=np.asarray(split, 'U8'),
                   malig_prob=mp)
    if extra:
        payload.update(extra)
    np.savez_compressed(os.path.join(OUT_DIR, 'features_%s.npz' % tag), **payload)
    pd.DataFrame({'image_id': [str(v) for v in image_id], 'malignant': malignant,
                  'group': group, 'split': split}).to_parquet(
        os.path.join(OUT_DIR, 'meta_%s.parquet' % tag))

if ENCODE:
    # --- Fitzpatrick17k ---
    CSV_URL = 'https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv'
    fcsv = find_one('fitzpatrick17k.csv')
    f = pd.read_csv(fcsv) if fcsv else pd.read_csv(CSV_URL)
    print('fitz csv rows:', len(f))

    part_col = 'three_partition_label' if 'three_partition_label' in f.columns else 'nine_partition_label'
    tone_col = next(c for c in ['fitzpatrick_scale', 'fitzpatrick', 'fitzpatrick_centaur']
                    if c in f.columns)
    id_col = next(c for c in ['md5hash', 'hasher', 'image', 'filename'] if c in f.columns)

    # fixed filter and row order: later steps rely on them
    f = f[f[part_col].notna() & f[tone_col].notna()].reset_index(drop=True)
    print('after filter:', len(f))

    # images come from the zip; loose files on disk are the fallback
    zpath = find_one('fitzpatrick17k.zip')
    srcs, zf = [], None
    if zpath:
        zf = zipfile.ZipFile(zpath)
        zindex = {os.path.splitext(os.path.basename(n))[0]: n
                  for n in zf.namelist() if n.lower().endswith(('.jpg', '.jpeg', '.png'))}
        print('zip: %s | %d images indexed' % (zpath, len(zindex)))
        srcs = [((zf, zindex[str(h)]) if str(h) in zindex else '') for h in f[id_col]]
    else:
        index = {}
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            for p in glob.glob(os.path.join(DATA_DIR, '**', ext), recursive=True):
                index.setdefault(os.path.splitext(os.path.basename(p))[0], p)
        srcs = [index.get(str(h), '') for h in f[id_col]]
    print('images located: %d / %d' % (sum(1 for s in srcs if s), len(srcs)))
    assert sum(1 for s in srcs if s) > 0.95 * len(srcs), 'too many images missing'


    def tone_group(v):
        try:
            v = float(v)
        except Exception:
            return 'unknown'
        if v in (1, 2):
            return 'light'
        if v in (3, 4):
            return 'mid'
        if v in (5, 6):
            return 'dark'
        return 'unknown'


    malignant = (f[part_col].astype(str) == 'malignant').astype(int).values
    group = [tone_group(v) for v in f[tone_col]]
    rng = np.random.default_rng(0)
    u = rng.random(len(f))
    split = np.where(u < 0.70, 'train', np.where(u < 0.85, 'val', 'test'))

    cache_dataset('fitz', f[id_col].astype(str).tolist(), srcs, malignant, group, split,
                  extra={'three_partition_label': f[part_col].astype(str).values.astype('U32')})
    print('malignant:', int(malignant.sum()), '/', len(malignant),
          '| by tone:', {g: group.count(g) for g in ['light', 'mid', 'dark', 'unknown']})

if ENCODE:
    # --- DDI ---
    dcsv = find_one('ddi_metadata.csv')
    if not dcsv:
        print('DDI metadata not found - attach the DDI dataset and re-run this cell.')
    else:
        dd = pd.read_csv(dcsv)  # keep CSV row order: later steps join on it
        file_col = next(c for c in ['DDI_file', 'file', 'filename', 'image'] if c in dd.columns)
        ddir = os.path.dirname(dcsv)
        srcs = []
        for fn in dd[file_col]:
            cand = os.path.join(ddir, str(fn))
            srcs.append(cand if os.path.exists(cand) else (find_one(str(fn), [ddir]) or ''))
        tone2g = {12: 'light', 34: 'mid', 56: 'dark'}
        group = [tone2g.get(int(v), 'unknown') for v in dd['skin_tone']]
        malignant = dd['malignant'].astype(int).values
        cache_dataset('ddi', dd[file_col].astype(str).tolist(), srcs, malignant, group,
                      ['test'] * len(dd),
                      extra={'disease': dd['disease'].astype(str).values.astype('U64')})
        print('ddi:', {g: group.count(g) for g in ['light', 'mid', 'dark']},
              '| malignant', int(malignant.sum()))
        # later steps join the SkinCon DDI labels on this file's row order
        shutil.copyfile(dcsv, os.path.join(os.path.abspath(args.out_dir), 'ddi_metadata.csv'))

if ENCODE:
    # --- Derm7pt: clinical (emb_clinic) and dermoscopic (emb_derm) images ---
    mcsv = find_one('meta.csv') or find_one('derm7pt_meta.csv')
    if not mcsv:
        print('Derm7pt metadata not found - attach the Derm7pt dataset and re-run this cell.')
    else:
        m = pd.read_csv(mcsv)
        root = os.path.dirname(mcsv)

        def resolve(col):
            out = []
            for v in m[col]:
                cand = os.path.join(root, 'images', str(v))
                out.append(cand if os.path.exists(cand)
                           else (find_one(os.path.basename(str(v)), [root]) or ''))
            return out

        mel = m['diagnosis'].astype(str).str.lower().str.contains('melanoma').astype(int).values
        emb_c, mask_c = encode_images(resolve('clinic'))
        emb_d, mask_d = encode_images(resolve('derm'))
        cs_c, _ = concept_scores_zs(emb_c, mask_c)
        cs_d, _ = concept_scores_zs(emb_d, mask_d)
        np.savez_compressed(
            os.path.join(OUT_DIR, 'features_derm7pt.npz'),
            image_id=np.asarray([str(v) for v in m['case_num']], dtype=object),
            emb_clinic=emb_c, mask_clinic=mask_c, emb_derm=emb_d, mask_derm=mask_d,
            concept_scores_clinic=cs_c, concept_scores_derm=cs_d,
            mel_label=mel.astype(np.int64),
            diagnosis=m['diagnosis'].astype(str).values.astype('U64'))
        print('derm7pt:', len(m), 'cases |', int(mel.sum()), 'melanoma')

# --- manifest and image_id check, also for a reused cache ---
man = {'model': MODEL_NAME, 'concept_names': list(CONCEPT_PROMPTS),
       'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'image_id_stored': True,
       'encoded_this_run': bool(ENCODE),
       'note': ('image_id is stored in every cache, so downstream joins are exact by '
                'construction and need no row-order reconstruction.')}
json.dump(man, open(os.path.join(OUT_DIR, 'manifest.json'), 'w'), indent=1)

for fn in sorted(os.listdir(OUT_DIR)):
    if not fn.endswith('.npz'):
        continue
    z = np.load(os.path.join(OUT_DIR, fn), allow_pickle=True)
    assert 'image_id' in z.files, fn + ' has no image_id - the cache contract requires it'
    ids = [str(v) for v in z['image_id']]
    print('%-28s rows %-7d unique image_id %d' % (fn, len(ids), len(set(ids))))
if zf is not None:
    zf.close()
    zf = None
print('\ndone. All caches carry image_id; run_pipeline.py reads them from', OUT_DIR)
