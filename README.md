# Point-FCW: a TDA machine learning pipeline for point cloud classification
Point-FCW: transposed-FCW Graph Representation for Point Cloud Classification using TDA

---

# Point-FCW

![License Badge](https://img.shields.io/github/license/hawkinglai/Point-FCW)
![GitHub Stars](https://img.shields.io/github/stars/hawkinglai/Point-FCW)

Point-FCW is an open-source project for point cloud classification using **transposed fully connected and weighted (t-FCW)** graph representation integrated with topological data analysis (TDA). Our lightweight framework efficiently extracts features from 3D point clouds, offering a robust and complementary approach to state-of-the-art models.

---

## 🔥 Key Features
- **t-FCW Graph Representation**: A highly efficient and scalable transformation for point clouds, reducing computational complexity.
- **Topological Data Analysis (TDA)**: Feature extraction via persistence diagrams, persistence landscapes, Betti curves, and more.
- **Model Integration**: Enhance performance by plugging Point-FCW into existing networks like Point-NN or PointMLP.
- **High Efficiency**: Handles large point cloud volumes with stable computational complexity.
- **Visualization**: Enables matching score landscapes for classification progress observation.

---

## 🚀 Performance Highlights

| Model               | Dataset       | Accuracy (%) |
|---------------------|---------------|--------------|
| Point-FCW           | ModelNet40    | 75.28        |
| Point-FCW + PointMLP | OBJ-BG       | 91.57        |
| Point-FCW + PointNN | PB-T50-RS    | 65.65        |

---

## 📖 How It Works

1. **t-FCW Graph Construction**: 
   - Transforms point cloud into a transposed fully connected and weighted graph.
   - Efficiently extracts meaningful patterns while reducing computational demands.
   - See pseudocode implementation [here](docs/t-FCW.md).

2. **Feature Extraction**:
   - Applies TDA via vectorization techniques: persistence entropy, landscapes, Betti curves, etc.
   - Ensures lightweight yet effective feature representation.

3. **Model Integration**:
   - Use Point-FCW as a standalone classifier or integrate it with neural networks for enhanced performance.

---

## 🛠️ Installation

To get started, clone the repository and install dependencies:

```bash
git clone https://github.com/hawkinglai/Point-FCW.git
cd Point-FCW
pip install -r requirements.txt
```

---

## 📂 Usage

### Example Workflow

```python
from point_fcw import PointFCW

# Load and preprocess point cloud data
point_cloud = load_point_cloud('example.pcd')

# Create t-FCW graph representation
graph = PointFCW.to_tfcw(point_cloud)

# Extract features using TDA
features = PointFCW.extract_tda_features(graph)

# Classify using non-parametric head
label = PointFCW.classify(features)
```

Check out the [tutorials](docs/tutorials.md) for detailed examples.

---

## 📊 Benchmark Datasets

The following datasets were used for evaluation:
- [ModelNet40](https://modelnet.cs.princeton.edu/)
- [ScanObjectNN](https://hkust-vgd.github.io/scanobjectnn/)
  - Sub-benchmarks: OBJ-BG, PB-T50-RS

---

## 🤝 Contributing

We welcome contributions from the community! To contribute:
1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature`.
3. Commit your changes: `git commit -m 'Add your feature'`.
4. Push the branch: `git push origin feature/your-feature`.
5. Open a pull request.

---

## 📄 Citation

If you use Point-FCW in your research, please cite the following paper:

```bibtex
@article{lai2025pointfcw,
  title={Point-FCW: transposed-FCW Graph Representation for Point Cloud Classification using TDA},
  author={Lai, Haijian and Liu, Bowen and Lam, Chan-Tong and Ng, Benjamin and Im, Sio-Kei},
  journal={Journal of LATEX Class Files},
  volume={14},
  number={8},
  pages={1--5},
  year={2025}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For any inquiries or support, please contact:
- Haijian Lai: [haijian.lai@mpu.edu.mo](mailto:haijian.lai@mpu.edu.mo)

---

Feel free to let me know what sections need tweaking, adding, or removing!
