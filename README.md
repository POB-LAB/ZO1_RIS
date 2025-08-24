# ZO-1 Network Analysis Tool (Deployment Version)

## Overview
This is a deployment-ready version of the ZO-1 Network Analysis Tool, optimized for Streamlit Cloud deployment. The tool performs AI-powered cell segmentation and network analysis using the Radial Integrity Score (RIS) method.

## Key Features
- **AI-Powered Segmentation**: Uses Cellpose for automatic cell detection
- **RIS Analysis**: Implements Radial Integrity Score for network quantification
- **CPU-Optimized**: Designed for deployment without GPU requirements
- **Smart Resampling**: Default 512x512 processing with option for full resolution
- **Export Functionality**: CSV and detailed report downloads
- **Classic Segmentation Toolkit**: Otsu thresholding by default with adjustable strength, optional ridge filtering, watershed-based labelling and skeletonisation with configurable thickness

## 🚀 **Live Demo**
Access the deployed application at: [Your Streamlit Cloud URL]

## 📋 **Prerequisites**
- Modern web browser (Chrome, Firefox, Safari, Edge)
- No local installation required - runs entirely in the cloud
- Supports common image formats: TIF, TIFF, PNG, JPG, JPEG

## 🖥️ **System Requirements**
- **Deployment**: Streamlit Cloud (CPU-only)
- **Memory**: Optimized for cloud deployment
- **Processing**: CPU-based AI segmentation
- **Storage**: Temporary image processing (no data stored)

## Deployment
This version is optimized for Streamlit Cloud deployment with:
- CPU-only processing for maximum compatibility
- Optimized memory usage
- Streamlined dependencies
- Deployment-ready configuration

## Contact
For batch operations or technical support, contact: pierre.bagnaninchi@ed.ac.uk

## Usage
1. Upload a ZO-1 fluorescence image
2. Adjust cell diameter estimate if needed
3. Run segmentation
4. Run analysis
5. Download results

## Technical Notes
- Default image resampling to 512x512 for faster processing
- Full resolution option available (but slower)
- AI contour validation to remove phantom boundaries
- Automatic scaling compensation for RIS calculations

## 🔧 **Local Development** (Optional)
If you want to run this locally:

```bash
# Clone the repository
git clone [your-repo-url]
cd zo1_ris

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

## 📊 **Performance Notes**
- **Fast Mode**: 512x512 resampling for quick results
- **Full Resolution**: Original image size (slower but more detailed)
- **CPU Processing**: Optimized for cloud deployment
- **Memory Efficient**: Streamlined for Streamlit Cloud constraints

## 🆘 **Troubleshooting**
- **Slow Processing**: Use 512x512 resampling option
- **Large Images**: Consider downsampling for better performance
- **Browser Issues**: Try refreshing or using a different browser
- **Contact Support**: pierre.bagnaninchi@ed.ac.uk for technical issues
