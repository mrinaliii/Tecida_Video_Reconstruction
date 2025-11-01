# Video Frame Reconstruction Challenge

##  Overview
A high-performance solution for reconstructing jumbled video frames using computer vision and optimization algorithms.  
This tool efficiently restores the original temporal sequence of shuffled video frames through advanced similarity analysis and parallel processing.

---

## Algorithm Approach

### Core Methodology
The reconstruction employs a similarity-based graph approach that leverages the principle of temporal coherence in video sequences.  
Consecutive frames in natural video exhibit minimal visual changes, allowing us to reconstruct the original sequence by identifying frames with maximum similarity.

---

## Technical Implementation

- **Similarity Analysis:** Uses *structural similarity* and *absolute difference* metrics on downscaled frame features.  
- **Graph Construction:** Frames represented as nodes in a complete graph with edge weights based on *dissimilarity scores*.  
- **Optimal Sequencing:** *Minimum Spanning Tree (MST)* algorithm finds the path connecting most similar frames.  
- **Parallel Processing:** *Multi-threaded computation* optimized for modern multi-core processors.

---

## Performance Characteristics

- **Accuracy:** High-quality reconstruction using full similarity analysis  
- **Speed:** 30–60 seconds for 300 frames on standard hardware  
- **Scalability:** Efficient algorithms suitable for various frame counts  
- **Parallelization:** Utilizes all available CPU cores for optimal performance  

---

## Installation & Setup

###  Prerequisites
- Python **3.12+**  
- **4GB+** RAM  
- **1GB+** free disk space  

###  Dependencies
```bash
pip install -r requirements.txt
```

---

## Project Structure
<img width="419" height="168" alt="image" src="https://github.com/user-attachments/assets/baca2569-89f5-4479-a06d-07743c961b41" />


---

##  Usage Instructions

###  Quick Start Guide

1) Place input video in the data folder
2) Make sure you cd to the project's root directory
3) Create a python virtual environment
  ```bash
python -m venv venv
```
4) Activate the virtual environment
  ```bash
.\venv\Scripts\Activate
```
(Given command is for windows powershell)

5) Install Dependencies
```bash
pip install -r requirements.txt
```
6) Run by typing this in the terminal make sure your file extension is cd../project
```bash
python src/main.py
```

### Check Results
Output: output/reconstructed_video.mp4
Performance Log: execution_time.log

---

## Performance Results

### Actual Performance on my System:
 <img width="941" height="242" alt="image" src="https://github.com/user-attachments/assets/1d91cc86-556f-4ad8-83b7-b9e9fbbfeefa" />

---

## Evaluation Metrics

The solution optimizes for:
1) Frame-wise Similarity: Structural preservation between consecutive frames
2) Temporal Coherence: Smooth transitions and natural motion flow
3) Computational Efficiency: Parallel processing and algorithmic optimization
4) Resource Utilization: Memory-efficient frame processing

---

## Technical Specifications
1) Input Format: MP4 video, 30 FPS, 300 frames (10 seconds)
2) Output Format: MP4 video with restored temporal sequence
3) Processing: Multi-threaded, utilizes all available CPU cores
4) Memory: Efficient batch processing for large frame counts

---

## Support

### If you encounter issues:

1) Verify that the video meets specifications (300 frames, 30 FPS)
2) Check that all dependencies are installed
3) Ensure sufficient disk space for temporary files
4) Review execution_time.log for performance metrics

---






