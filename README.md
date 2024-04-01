# Deep Learning Solution for Precise Subtitle Segmentation
Anush Veeranala, Xinyu Zhang, Lilin Jin

## Overview

This project develops a reliable, deep learning-based subtitle segmentation solution 
to improve video accessibility and comprehension. Our work focuses on leveraging 
advanced neural network architectures to accurately segment subtitles, ensuring 
they are properly aligned with spoken content and displayed coherently on the screen. 
This solution is particularly aimed at academic institutions, conferences, and any 
scenario where resources for proprietary subtitle segmentation tools are limited.

## Features

- **Subtitle Segmentation**: Splits transcribed text into appropriately timed subtitle
  blocks and lines using deep learning models.
- **Multi-Language Support** (unfinished): Designed to work across various languages,
  enhancing the accessibility of diverse video content.
- **User-friendly Tool** (unfinished): Offers a straightforward interface for
  uploading video files or YouTube links and receiving segmented subtitles.
- **Open Source**: Committed to the open-source community, all code and models are
  freely available for use and modification.

## Installation

To set up the project on your local machine, follow these steps:

1. Clone the repository.
2. Install required dependencies.
   ```
   pip install -r requirements.txt
   ```
3. Run the following command: 

4. Optional arguments are available for specifying output formats, languages, and
    other configurations. Avaliable arguments: 

## Data

Our models are trained on the MuST-Cinema dataset, enriched with markers for 
subtitle segmentation. The dataset is available under the CC BY-NC-ND 4.0 license 
and comprises multilingual speech translation triplets with special symbols for 
subtitle breaks.

## Evaluation

We employ a comprehensive evaluation framework considering various factors like 
on-screen duration, voice synchronization, and text readability. Our metrics 
include Sigma (segmentation quality), Break Coverage, and Length Conformity, 
ensuring our models' effectiveness across diverse content types.

## Contributions

We welcome contributions from the community. If you're interested in improving 
the project or have suggestions, please feel free to fork the repository, 
make changes, and submit a pull request.


## Acknowledgments

- Special thanks to the authors and contributors of the MuST-Cinema dataset.
- Our project builds on research from various sources, including advancements in neural text segmentation and unsupervised learning models.

