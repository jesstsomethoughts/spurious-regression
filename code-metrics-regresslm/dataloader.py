import numpy as np
import matplotlib as plt
import torch
import ast
from scipy.ndimage import convolve1d, gaussian_filter1d
from scipy.signal.windows import triang
from scipy.stats import wasserstein_distance
from sklearn.manifold import MDS
from torch.utils import data

'''
Raw CDSS Dataset (No Weighting)
'''
class CDSSDataset(data.Dataset):
    def __init__(self, inputs, targets, tokenizer, max_length=2048):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.inputs[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        target_tokens = self.tokenizer.floats_to_token_ids([self.targets[idx]])

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(target_tokens),
            'weight': torch.tensor(1.0, dtype=torch.float32)
        }

################################################################################################

'''
LDS Helper Function
'''
def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window

'''
Class for CDSS Dataset with Label Distribution Smoothing (LDS)
'''
class CDSSLDS(data.Dataset):
    def __init__(
        self,
        dataset,
        selected_indices,
        split="train",
        reweight="sqrt_inv",
        lds=True,
        lds_kernel="gaussian",
        lds_ks=5,
        lds_sigma=2,
        bin_width=1.0,
    ):
        self.split = split
        self.dataset = dataset
        self.selected_indices = list(selected_indices)
        self.bin_width = float(bin_width)

        self.targets = np.zeros(len(self.selected_indices), dtype=np.float64)
        self.langs = []

        for i, actual_index in enumerate(self.selected_indices):
            row = self.dataset[actual_index]

            metadata = row["metadata"]
            # a
            lang =  eval(metadata)["language"]

            # y
            target = float(row["target"])

            self.langs.append(lang)
            self.targets[i] = target

        self.langs = np.asarray(self.langs, dtype=object)

        self.weights = self._prepare_weights(
            reweight=reweight,
            lds=lds,
            lds_kernel=lds_kernel,
            lds_ks=lds_ks,
            lds_sigma=lds_sigma,
            bin_width=self.bin_width,
        )

    @staticmethod
    def _bin_index(y, y_min, num_bins, bin_width):
        if num_bins <= 1:
            return 0
        b = int(np.floor((float(y) - y_min) / bin_width))
        return max(0, min(num_bins - 1, b))

    @staticmethod
    def _num_bins(y_min, y_max, bin_width):
        if y_max <= y_min:
            return 1
        return int(np.floor((y_max - y_min) / bin_width)) + 1

    def _prepare_group_weights(
        self,
        labels,
        reweight,
        lds,
        lds_kernel,
        lds_ks,
        lds_sigma,
        bin_width,
        clip_min=5.0,
        clip_max=1000.0,
    ):
        labels = np.asarray(labels, dtype=np.float64).ravel()

        if len(labels) == 0:
            return np.array([], dtype=np.float32)

        if reweight == "none":
            return np.ones(len(labels), dtype=np.float32)

        y_min = float(np.min(labels))
        y_max = float(np.max(labels))
        num_bins = self._num_bins(y_min, y_max, bin_width)

        counts = np.zeros(num_bins, dtype=np.float64)
        bin_ids = np.zeros(len(labels), dtype=np.int64)

        for i, y in enumerate(labels):
            b = self._bin_index(y, y_min, num_bins, bin_width)
            bin_ids[i] = b
            counts[b] += 1.0
       
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(num_bins), counts, marker="o")
        plt.xlabel("Bin index")
        plt.ylabel("Raw count")
        plt.title("Raw counts before reweight")
        plt.tight_layout()
        plt.show()
        if reweight == "sqrt_inv":
            counts = np.sqrt(counts)
        elif reweight == "inverse":
            counts = np.clip(counts, clip_min, clip_max)
        else:
            raise ValueError(f"Unknown reweight: {reweight}")

        if lds:
            window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            counts = convolve1d(counts, weights=window, mode="constant")

        denom = np.maximum(counts[bin_ids], 1e-12)
        weights = (1.0 / denom).astype(np.float32)
        weights *= len(weights) / float(np.sum(weights))
        return weights

    def _prepare_weights(
        self,
        reweight,
        lds,
        lds_kernel,
        lds_ks,
        lds_sigma,
        bin_width,
    ):
        assert reweight in {"none", "inverse", "sqrt_inv"}
        if lds:
            assert reweight != "none", "Set reweight to 'sqrt_inv' or 'inverse' when using LDS"

        weights = np.zeros(len(self.selected_indices), dtype=np.float32)
        unique_langs = np.unique(self.langs)

        for lang in unique_langs:
            idx = np.where(self.langs == lang)[0]
            weights[idx] = self._prepare_group_weights(
                labels=self.targets[idx],
                reweight=reweight,
                lds=lds,
                lds_kernel=lds_kernel,
                lds_ks=lds_ks,
                lds_sigma=lds_sigma,
                bin_width=bin_width,
            )

        total = float(np.sum(weights))
        if total > 0:
            weights *= len(weights) / total

        return weights.tolist()

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, index):
        actual_index = self.selected_indices[index]
        row = self.dataset[actual_index]

        metadata = row["metadata"]
        
        # a
        lang =  eval(metadata)["language"]

        # x
        input_text = f"# CDSS\n# Language: {lang}\n{row['input']}"

        # y
        target = float(row["target"])
        attr = lang
        weight = self.weights[index]

        return index, input_text, target, attr, weight

############################################################################################

'''
MDS Helper Functions
'''
def _row_normalize(M, eps=1e-12):
    return M / (M.sum(axis=1, keepdims=True) + eps)


def _pairwise_wasserstein_attr_distance(y, a, unique_a):
    y = np.asarray(y).ravel().astype(np.float64)
    a = np.asarray(a).ravel()

    G = len(unique_a)
    samples = [y[a == g] for g in unique_a]

    D = np.zeros((G, G), dtype=np.float64)
    for i in range(G):
        for j in range(i + 1, G):
            d = wasserstein_distance(samples[i], samples[j])
            D[i, j] = D[j, i] = d
    return D


def build_attr_kernel_via_mds(y, a, unique_a, tau=None, knn=None, random_state=0, eps=1e-12):
    D = _pairwise_wasserstein_attr_distance(y, a, unique_a)

    try:
        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            metric=True,
            normalized_stress="auto",
            random_state=random_state,
            n_init=4,
            max_iter=300,
        )
    except TypeError:
        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            metric=True,
            random_state=random_state,
            n_init=4,
            max_iter=300,
        )

    Z = mds.fit_transform(D)
    stress = getattr(mds, "stress_", None)

    if tau is None:
        diff = Z[:, None, :] - Z[None, :, :]
        Dz = np.sqrt(np.sum(diff * diff, axis=-1))
        off = Dz[np.triu_indices(len(unique_a), k=1)]
        tau = float(np.median(off) + eps)

    diff = Z[:, None, :] - Z[None, :, :]
    Dz2 = np.sum(diff * diff, axis=-1)
    K = np.exp(-Dz2 / (2.0 * tau * tau))
    np.fill_diagonal(K, 1.0)

    G = K.shape[0]
    if knn is not None and knn < G:
        for i in range(G):
            keep = np.argsort(-K[i])[: knn + 1]
            mask = np.ones(G, dtype=bool)
            mask[keep] = False
            K[i, mask] = 0.0

    K = _row_normalize(K, eps=eps)
    return K, Z, D, stress, tau


'''
CDSSMDS Dataset (Multi-Dimensional Scaling (MDS) + LDS Weighting)
'''
class CDSSMDS(data.Dataset):
    def __init__(
        self,
        dataset,
        selected_indices,
        tokenizer,
        max_length=2048,
        split="train",
        reweight="sqrt_inv",
        lds=True,
        lds_kernel="gaussian",
        lds_ks=5,
        lds_sigma=2,
        bin_width=1.0,
        attr_mds=True,
        attr_mds_tau=None,
        attr_mds_knn=None,
        print_kernel=True,
        kernel_topk=10,
    ):
        self.split = split
        self.dataset = dataset
        self.selected_indices = list(selected_indices)
        self.bin_width = float(bin_width)
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.targets = np.zeros(len(self.selected_indices), dtype=np.float64)
        self.langs = []
        self.inputs = []

        for i, actual_index in enumerate(self.selected_indices):
            row = self.dataset[actual_index]
            metadata = row["metadata"]
            if isinstance(metadata, str):
                metadata = ast.literal_eval(metadata)

            lang = metadata["language"]
            target = float(row["target"])
            input_text = f"# CDSS\n# Language: {lang}\n{row['input']}"

            self.langs.append(lang)
            self.targets[i] = target
            self.inputs.append(input_text)

        self.langs = np.asarray(self.langs, dtype=object)

        self.weights = self._prepare_weights(
            reweight=reweight,
            lds=lds,
            lds_kernel=lds_kernel,
            lds_ks=lds_ks,
            lds_sigma=lds_sigma,
            bin_width=self.bin_width,
            attr_mds=attr_mds,
            attr_mds_tau=attr_mds_tau,
            attr_mds_knn=attr_mds_knn,
            print_kernel=print_kernel,
            kernel_topk=kernel_topk,
        )

    @staticmethod
    def _bin_index(y, y_min, num_bins, bin_width):
        if num_bins <= 1:
            return 0
        b = int(np.floor((float(y) - y_min) / bin_width))
        return max(0, min(num_bins - 1, b))

    @staticmethod
    def _num_bins(y_min, y_max, bin_width):
        if y_max <= y_min:
            return 1
        return int(np.floor((y_max - y_min) / bin_width)) + 1

    def _prepare_group_weights(
        self,
        labels,
        reweight,
        lds,
        lds_kernel,
        lds_ks,
        lds_sigma,
        bin_width,
        clip_min=5.0,
        clip_max=1000.0,
    ):
        labels = np.asarray(labels, dtype=np.float64).ravel()

        if len(labels) == 0:
            return np.array([], dtype=np.float32)

        if reweight == "none":
            return np.ones(len(labels), dtype=np.float32)

        y_min = float(np.min(labels))
        y_max = float(np.max(labels))
        num_bins = self._num_bins(y_min, y_max, bin_width)

        counts = np.zeros(num_bins, dtype=np.float64)
        bin_ids = np.zeros(len(labels), dtype=np.int64)

        for i, y in enumerate(labels):
            b = self._bin_index(y, y_min, num_bins, bin_width)
            bin_ids[i] = b
            counts[b] += 1.0

        if reweight == "sqrt_inv":
            counts = np.sqrt(counts)
        elif reweight == "inverse":
            counts = np.clip(counts, clip_min, clip_max)
        else:
            raise ValueError(f"Unknown reweight: {reweight}")

        if lds:
            window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            counts = convolve1d(counts, weights=window, mode="constant")

        denom = np.maximum(counts[bin_ids], 1e-12)
        weights = (1.0 / denom).astype(np.float32)
        weights *= len(weights) / float(np.sum(weights))
        return weights

    def _prepare_weights(
        self,
        reweight,
        lds,
        lds_kernel,
        lds_ks,
        lds_sigma,
        bin_width,
        attr_mds,
        attr_mds_tau,
        attr_mds_knn,
        print_kernel,
        kernel_topk,
    ):
        assert reweight in {"none", "inverse", "sqrt_inv"}
        if lds:
            assert reweight != "none", "Set reweight to 'sqrt_inv' or 'inverse' when using LDS"

        weights = np.zeros(len(self.selected_indices), dtype=np.float32)
        unique_langs = np.unique(self.langs)

        for lang in unique_langs:
            idx = np.where(self.langs == lang)[0]
            weights[idx] = self._prepare_group_weights(
                labels=self.targets[idx],
                reweight=reweight,
                lds=lds,
                lds_kernel=lds_kernel,
                lds_ks=lds_ks,
                lds_sigma=lds_sigma,
                bin_width=bin_width,
            )

        if attr_mds:
            K, Z, D, stress, tau_used = build_attr_kernel_via_mds(
                y=self.targets,
                a=self.langs,
                unique_a=unique_langs,
                tau=attr_mds_tau,
                knn=attr_mds_knn,
                random_state=0,
            )

            if print_kernel:
                print(f"[attr_mds] tau_used={tau_used:.6f}, stress={stress}")

            n_g = np.array([(self.langs == g).sum() for g in unique_langs], dtype=np.float64)
            n_smooth = K @ n_g
            n_smooth = np.maximum(n_smooth, 1e-12)

            if reweight == "sqrt_inv":
                factors = 1.0 / np.sqrt(n_smooth)
            elif reweight == "inverse":
                factors = 1.0 / n_smooth
            else:
                factors = np.ones_like(n_smooth)

            factors *= len(factors) / (factors.sum() + 1e-12)

            print("[attr_mds] group stats: lang, n_raw, n_smooth, factor")
            for j, lang in enumerate(unique_langs):
                print(
                    f"  lang={lang}  n={int(n_g[j])}  "
                    f"n_smooth={n_smooth[j]:.2f}  factor={factors[j]:.4f}"
                )

            for j, lang in enumerate(unique_langs):
                idx = np.where(self.langs == lang)[0]
                weights[idx] *= float(factors[j])

        total = float(np.sum(weights))
        if total > 0:
            weights *= len(weights) / total

        return weights.tolist()

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, index):
        encoding = self.tokenizer(
            self.inputs[index],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        target_tokens = self.tokenizer.floats_to_token_ids([self.targets[index]])

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(target_tokens),
            'weight': torch.tensor(self.weights[index], dtype=torch.float32)
        }
