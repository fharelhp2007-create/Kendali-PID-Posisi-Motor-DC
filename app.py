import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# KONFIGURASI HALAMAN UTAMA
# ==========================================
st.set_page_config(
    page_title="Simulasi Sistem Kendali PID - Motor DC", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Gaya CSS untuk Mempercantik Tampilan Card
st.markdown("""
    <style>
    .metric-box {
        background-color: #1e222b;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #00ffcc;
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 14px;
        color: #a0aab2;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# Header Aplikasi
st.title("⚡ Simulasi Kendali PID Posisi Motor DC")
st.markdown("Ubah parameter fisik motor dan parameter kontroler PID untuk melihat respons transien posisi poros.")
st.markdown("---")

# ==========================================
# SIDEBAR: PARAMETER INPUT
# ==========================================
st.sidebar.header("🛠️ 1. Parameter Fisik Motor")
J = st.sidebar.slider("Momen Inersia (J)", 0.01, 0.50, 0.02, 0.01, format="%.2f kg.m²")
b = st.sidebar.slider("Redaman Mekanik (b)", 0.01, 0.50, 0.10, 0.01, format="%.2f N.m.s")
R = st.sidebar.slider("Resistansi (R)", 0.1, 5.0, 1.0, 0.1, format="%.1f Ohm")
L = st.sidebar.slider("Induktansi (L)", 0.1, 1.0, 0.5, 0.1, format="%.1f Henry")
K = st.sidebar.slider("Konstanta Motor (K)", 0.01, 0.50, 0.10, 0.01, format="%.2f V/rad/s")

st.sidebar.header("🎛️ 2. Parameter Kontroler PID")
Kp = st.sidebar.slider("Proporsional (Kp)", 0.0, 50.0, 15.0, 0.5)
Ki = st.sidebar.slider("Integral (Ki)", 0.0, 20.0, 5.0, 0.1)
Kd = st.sidebar.slider("Derivatif (Kd)", 0.0, 10.0, 2.5, 0.1)

# ==========================================
# FUNGSI SIMULASI NUMERIK DENGAN KENDALI PID
# ==========================================
@st.cache_data
def simulate_motor_pid(J_val, b_val, R_val, L_val, K_val, Kp_val, Ki_val, Kd_val):
    dt = 0.002
    time = np.arange(0, 5.0, dt) # Simulasi 5 detik sudah cukup terlihat stabilnya
    
    theta = np.zeros_like(time)
    setpoint = np.ones_like(time) # Target posisi = 1.0 radian
    
    # Kondisi awal keadaan diam
    i_val, w_val, t_val = 0.0, 0.0, 0.0
    integral_error = 0.0
    prev_error = 1.0
    
    for i in range(1, len(time)):
        # 1. Hitung Error Posisi
        error = 1.0 - t_val
        integral_error += error * dt
        derivative_error = (error - prev_error) / dt
        
        # 2. Sinyal Kendali PID sebagai Tegangan Masukan (V)
        V = (Kp_val * error) + (Ki_val * integral_error) + (Kd_val * derivative_error)
        
        # Batasan tegangan fisik (Saturasi) misal maks 12V dan min -12V
        V = max(min(V, 12.0), -12.0)
        
        # 3. Persamaan Diferensial Motor DC
        di_dt = (V - R_val * i_val - K_val * w_val) / L_val
        dw_dt = (K_val * i_val - b_val * w_val) / J_val
        
        # 4. Integrasi Euler
        i_val += di_dt * dt
        w_val += dw_dt * dt
        t_val += w_val * dt
        
        # Simpan nilai
        theta[i] = t_val
        prev_error = error
        
    return time, theta, setpoint

# Jalankan Komputasi Matematika
t, response, sp = simulate_motor_pid(J, b, R, L, K, Kp, Ki, Kd)

# ==========================================
# LAYOUT UTAMA (KOLOM)
# ==========================================
col_grafik, col_info = st.columns([3, 1])

with col_grafik:
    st.subheader("📈 Grafik Kinerja Kontrol Posisi")
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 4.8))
    
    ax.plot(t, response, lw=2.5, color='#00ffcc', label='Posisi Poros Aktual (rad)')
    ax.plot(t, sp, lw=1.5, color='#ff3366', linestyle='--', label='Target / Setpoint (1.0 rad)')
    
    ax.set_xlabel('Waktu (detik)', fontsize=10, color='#a0aab2')
    ax.set_ylabel('Posisi Sudut (rad)', fontsize=10, color='#a0aab2')
    ax.set_ylim(0, 2.0) # Mengunci batas vertikal agar visualisasi gerakan terlihat jelas
    ax.grid(True, linestyle=':', alpha=0.4, color='#4f5b66')
    ax.legend(loc='lower right', frameon=True, facecolor='#1e222b', edgecolor='none')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#4f5b66')
    ax.spines['bottom'].set_color('#4f5b66')
    
    st.pyplot(fig)

with col_info:
    st.subheader("📊 Analisis Sistem")
    st.markdown("Status sistem saat ini:")
    
    final_value = response[-1]
    
    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">🏁 Nilai Akhir (Steady State)</div>
            <div class="metric-value">{final_value:.4f} rad</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Deteksi apakah sistem berhasil stabil mendekati target (error < 5%)
    status_sistem = "STABIL" if abs(1.0 - final_value) < 0.05 else "BELUM STABIL"
    warna_status = "#00ffcc" if status_sistem == "STABIL" else "#ff3366"
    
    st.markdown(f"""
        <div class="metric-box" style="border-left-color: {warna_status};">
            <div class="metric-title">📢 Status Kendali</div>
            <div class="metric-value" style="color: {warna_status};">{status_sistem}</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="metric-box" style="border-left-color: #ffcc00;">
            <div class="metric-title">🎛️ Mode Operasi</div>
            <div class="metric-value">Closed-Loop PID</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Dikembangkan untuk melengkapi lampiran Tugas Proyek Persamaan Diferensial | 2026")