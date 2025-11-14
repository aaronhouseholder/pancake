# Pancake
**Pipeline for Atmospheric Narrow-band/Cross-correlation Analysis with KEck-KPF**

pancake is designed to reduce, clean, and analyze Keck/KPF data for atmospheric signals using both narrow-band and cross-correlation techniques. It performs:

- Blaze removal  
- Continuum normalization  
- Order stitching  
- Science Fiber combination  
- Telluric correction  
- Atmospheric detection

To use the repo:

```
git clone https://github.com/aaronhouseholder/pancake/
cd pancake
conda create -n pancake python=3.9
conda activate pancake
conda install numpy scipy astropy pandas matplotlib tqdm
pip install wotan
pip install lmfit
pip install jupyter
```
