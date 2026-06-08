# -*- coding: utf-8 -*-
"""
Aplikasi Utama: Analisis Intensitas Emosi Instagram
(Integrasi MTCNN, Dual-Branch CNN, ANN, & ANFIS 6-Input + Dukungan XAI)
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, resnet50
from torchvision import transforms
from mtcnn import MTCNN
from huggingface_hub import hf_hub_download

# ==========================================
# 0. PENGATURAN SISTEM & LABEL
# ==========================================
KELAS_EMOSI = ['Senang (Amusement)', 'Marah (Anger)', 'Netral (Contentment)', 'Sedih (Sadness)']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Sistem Berjalan di: {device}")

# Dinamisasi path agar aman saat di-deploy ke cloud server (Linux/Windows)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ==========================================
# 1. DEFINISI ARSITEKTUR KELAS (WAJIB ADA UNTUK LOAD BOBOT)
# ==========================================
# A. Ekstraktor ResNet 256d
class ResNet256Extractor(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.fc_256d = base_model.fc[0]
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc_256d(x)

# B. Modul ANN (Fusion + Refiner)
class DualBranchANN(nn.Module):
    def __init__(self):
        super(DualBranchANN, self).__init__()
        self.weight_face = nn.Parameter(torch.tensor(1.0))
        self.weight_scene = nn.Parameter(torch.tensor(1.0))
        self.fusion_fc = nn.Linear(544, 256)
        self.fusion_bn = nn.BatchNorm1d(256)
        self.fusion_relu = nn.ReLU()
        self.fusion_dropout = nn.Dropout(0.3)
        self.hidden1 = nn.Linear(256, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.hidden2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.output_layer = nn.Linear(64, 4)

    def forward(self, face, scene):
        w_face = face * self.weight_face
        w_scene = scene * self.weight_scene
        concat_feat = torch.cat((w_face, w_scene), dim=1)
        x = self.fusion_dropout(self.fusion_relu(self.fusion_bn(self.fusion_fc(concat_feat))))
        x = self.dropout1(self.relu1(self.hidden1(x)))
        x = self.dropout2(self.relu2(self.hidden2(x)))
        return self.output_layer(x)

# C. Modul ANFIS (Neuro-Fuzzy 6-Input)
class PyTorchANFIS(nn.Module):
    def __init__(self, num_inputs=6, num_mfs=3, num_classes=4):
        super(PyTorchANFIS, self).__init__()
        self.mu = nn.Parameter(torch.rand(num_inputs, num_mfs))
        self.sigma = nn.Parameter(torch.ones(num_inputs, num_mfs) * 0.5)
        self.rule_weights = nn.Linear(num_inputs * num_mfs, num_classes)

    def forward(self, x):
        x_expanded = x.unsqueeze(2) 
        mf_out = torch.exp(-0.5 * ((x_expanded - self.mu) / self.sigma)**2)
        fuzzy_degrees = mf_out.view(x.size(0), -1)
        out_crisp = self.rule_weights(fuzzy_degrees)
        return torch.sigmoid(out_crisp)

# ==========================================
# 2. INISIALISASI & MUAT MODEL VIA HF (PUBLIC)
# ==========================================
print("🔄 Memuat Detektor Wajah (MTCNN)...")
face_detector = MTCNN()

print("🔄 Mengunduh & Memuat Model dari Hugging Face (Public)...")

# Helper: Download dari Hugging Face
def get_model_path(repo_id, filename):
    return hf_hub_download(repo_id=repo_id, filename=filename)

# A. Face Branch
face_extractor = efficientnet_b0(weights=None)
face_extractor.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(1280, 7))
path_face = get_model_path("bun1110/efficientnet-fer2013", "efficientnet_b0_fer2013_best_weights (1).pth")
face_extractor.load_state_dict(torch.load(path_face, map_location=device))
face_extractor.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(1280, 256))
face_extractor.to(device).eval()

# B. Scene Branch
base_resnet = resnet50(weights=None)
base_resnet.fc = nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, 4))
scene_extractor = ResNet256Extractor(base_resnet)
path_scene = get_model_path("bun1110/resnet-emoset", "resnet50_scene_biro_256d.pth")
scene_extractor.load_state_dict(torch.load(path_scene, map_location=device))
scene_extractor.to(device).eval()

# C. ANN
ann_model = DualBranchANN().to(device)
path_ann = get_model_path("bun1110/modul-ann", "modul_ann_terbaik.pth")
ann_model.load_state_dict(torch.load(path_ann, map_location=device))
ann_model.eval()

# D. ANFIS
anfis_model = PyTorchANFIS(num_inputs=6).to(device)
path_anfis = get_model_path("bun1110/modul-anfis", "modul_anfis_terbaik.pth")
anfis_model.load_state_dict(torch.load(path_anfis, map_location=device))
anfis_model.eval()

# ✅ BAGIAN YANG HILANG DITAMBAHKAN KEMBALI DI SINI
# Standarisasi Gambar
transform_img = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. FUNGSI PEMROSESAN INTI
# ==========================================
def extract_hsv_features(img_resized):
    hsv_img = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
    hist_h = cv2.calcHist([hsv_img], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv_img], [1], None, [8], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv_img], [2], None, [8], [0, 256]).flatten()
    hist_h /= (hist_h.sum() + 1e-7)
    hist_s /= (hist_s.sum() + 1e-7)
    hist_v /= (hist_v.sum() + 1e-7)
    return np.concatenate([hist_h, hist_s, hist_v])

def analyze_single_image(image_path):
    print(f"\n📸 Memproses Gambar: {os.path.basename(image_path)}")
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Gambar tidak dapat dibaca."
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    # 1. Ekstraksi Wajah (256d)
    hasil_deteksi = face_detector.detect_faces(img_rgb)
    if len(hasil_deteksi) > 0:
        print("🧑 Wajah Terdeteksi! Memproses ekspresi...")
        x, y, w, h = hasil_deteksi[0]['box']
        face_crop = img_rgb[max(0, y):y+h, max(0, x):x+w]
        face_tensor = transform_img(Image.fromarray(face_crop)).unsqueeze(0).to(device)
        with torch.no_grad():
            face_feat = face_extractor(face_tensor)
    else:
        print("🏞️ Wajah Tidak Terdeteksi. Menggunakan Zero-Padding.")
        face_feat = torch.zeros((1, 256)).to(device)
        
    # 2. Ekstraksi Suasana (288d)
    print("🎨 Membaca konteks suasana dan warna...")
    hsv_32d = extract_hsv_features(img_resized)
    img_tensor = transform_img(Image.fromarray(img_resized)).unsqueeze(0).to(device)
    with torch.no_grad():
        resnet_256d = scene_extractor(img_tensor)
        
    hsv_tensor = torch.tensor(hsv_32d, dtype=torch.float32).unsqueeze(0).to(device)
    scene_feat = torch.cat((resnet_256d, hsv_tensor), dim=1)
    
    # 3. Penalaran ANN (Mencari Probabilitas)
    print("🧠 Menganalisis jaringan saraf (ANN)...")
    with torch.no_grad():
        ann_logits = ann_model(face_feat, scene_feat)
        ann_probs = nn.Softmax(dim=1)(ann_logits)
        
    # --- PERSIAPAN 6-INPUT UNTUK ANFIS ---
    sat_hist = hsv_32d[16:24]
    val_hist = hsv_32d[24:32]
    
    bin_centers = np.array([0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375])
    avg_sat = np.sum(sat_hist * bin_centers)
    avg_val = np.sum(val_hist * bin_centers)
    
    context_tensor = torch.tensor([[avg_val, avg_sat]], dtype=torch.float32).to(device)
    anfis_input = torch.cat((ann_probs, context_tensor), dim=1)
        
    # 4. Penghalusan ANFIS (Skor Akhir) & XAI
    print("⚖️ Menghitung skor logika Fuzzy (ANFIS 6-Input)...")
    with torch.no_grad():
        anfis_scores = anfis_model(anfis_input)[0].cpu().numpy()
        # EKSTRAKSI PARAMETER UNTUK XAI STREAMLIT
        mu_vals = anfis_model.mu.cpu().numpy()
        sigma_vals = anfis_model.sigma.cpu().numpy()
        input_vals = anfis_input[0].cpu().numpy()
        
    # --- MENYUSUN HASIL ---
    print("\n" + "="*40)
    print("🏆 HASIL INTENSITAS EMOSI AKHIR")
    print("="*40)
    
    hasil_dict = {}
    for i, kelas in enumerate(KELAS_EMOSI):
        skor = anfis_scores[i]
        hasil_dict[kelas] = skor
        print(f"🔸 {kelas:<22}: {skor:.4f} ({skor*100:.1f}%)")
        
    emosi_dominan = max(hasil_dict, key=hasil_dict.get)
    print("-" * 40)
    print(f"🌟 KESIMPULAN: Emosi dominan pada foto ini adalah {emosi_dominan.upper()}!")
    print(f"🎨 Visual Konteks - Kecerahan: {avg_val*100:.1f}% | Saturasi: {avg_sat*100:.1f}%")
    print("=" * 40)
    
    # MENGEMBALIKAN DICTIONARY KOMPREHENSIF UNTUK VISUALISASI XAI
    return {
        "scores": hasil_dict,
        "inputs": input_vals,
        "mu": mu_vals,
        "sigma": sigma_vals
    }

# ==========================================
# 4. JALANKAN UJI COBA LOCAL
# ==========================================
if __name__ == "__main__":
    FOTO_TES = os.path.join(BASE_DIR, "foto_ujian2.jpg")
    
    if os.path.exists(FOTO_TES):
        hasil = analyze_single_image(FOTO_TES)
    else:
        print(f"\n⚠️ File '{FOTO_TES}' tidak ditemukan di folder ini.")