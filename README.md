# Point-FCW: a TDA machine learning pipeline for point cloud classification
Point-FCW: transposed-FCW Graph Representation for Point Cloud Classification using TDA

Point-FCW is a machine learning pipeline for point cloud classification using **transposed fully connected and weighted (t-FCW)** graph representation integrated with topological data analysis (TDA). Our lightweight framework efficiently extracts features from 3D point clouds, offering a robust and complementary approach to state-of-the-art models.

---

## 🔥 Key Features
- **t-FCW Graph Representation**: A highly efficient and scalable transformation for point clouds, reducing computational complexity.
- **Topological Data Analysis (TDA)**: Feature extraction via persistence diagrams, persistence landscapes, Betti curves, and more.
- **Model Integration**: Enhance performance by plugging Point-FCW into existing networks like Point-NN or PointMLP.

---

## 🛠️ Installation

To get started, clone the repository and install dependencies:

```bash
git clone https://github.com/hawkinglai/Point-FCW.git
cd Point-FCW
pip install -r requirements.txt
```

---

## 📄 Citation

If you use Point-FCW in your research, please cite the following paper:

```bibtex
@ARTICLE{10966157,
  author={Lai, Haijian and Liu, Bowen and Lam, Chan-Tong and Ng, Benjamin and Im, Sio-Kei},
  journal={IEEE Signal Processing Letters}, 
  title={Point-FCW: transposed-FCW Graph Representation for Point Cloud Classification using TDA}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Topological data analysis;machine learning;point cloud;non-parametric classification},
  doi={10.1109/LSP.2025.3561095}}
```
