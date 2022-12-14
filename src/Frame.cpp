#include "Frame.h"
#include "Settings.h"
#include <algorithm>
#include <utility>
#include <stdio.h> 
#include <stdlib.h> 
#include <iostream>
#include <iomanip>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "CycleTimer.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include "Sequence.h"
// For fast edge detection using structure forests
#include <opencv2/ximgproc.hpp>

using namespace cv;
namespace EdgeVO{

Frame::Frame()
    :m_image() , m_depthMap()
{}

Frame::Frame(Mat& image)
    :m_image(image) , m_depthMap( Mat() )
{}

Frame::Frame(std::string imagePath, std::string depthPath, Sequence* seq)
    :m_image(cv::imread(imagePath, cv::ImreadModes::IMREAD_GRAYSCALE)) , m_depthMap(cv::imread(depthPath, cv::ImreadModes::IMREAD_UNCHANGED)) , 
    m_imageName(imagePath), m_depthName(depthPath), m_seq(seq)
{
    //m_image.convertTo(m_image, CV_32FC1);
    //m_depthMap.convertTo(m_depthMap, CV_32FC1, EdgeVO::Settings::PIXEL_TO_METER_SCALE_FACTOR);
    m_pyramidImageUINT.resize(EdgeVO::Settings::PYRAMID_DEPTH);
    m_pyramidImage.resize(EdgeVO::Settings::PYRAMID_DEPTH);
    m_pyramidDepth.resize(EdgeVO::Settings::PYRAMID_DEPTH);
    m_pyramidEdge.resize(EdgeVO::Settings::PYRAMID_DEPTH);
    m_pyramid_Idx.resize(EdgeVO::Settings::PYRAMID_DEPTH);
    m_pyramid_Idy.resize(EdgeVO::Settings::PYRAMID_DEPTH);

    //> cchien3 add
    m_pyramidEdge_binary.resize(EdgeVO::Settings::PYRAMID_DEPTH);
    m_pyramidDT_dists.resize(EdgeVO::Settings::PYRAMID_DEPTH);
    m_pyramidDT_labels.resize(EdgeVO::Settings::PYRAMID_DEPTH);

    m_pyramidImageUINT[0] = m_image.clone(); 
    m_pyramidImage[0] = m_image;
    m_pyramidImage[0].convertTo(m_pyramidImage[0], CV_32FC1);
    m_pyramidDepth[0] = m_depthMap;
    m_pyramidDepth[0].convertTo(m_pyramidDepth[0], CV_32FC1, EdgeVO::Settings::PIXEL_TO_METER_SCALE_FACTOR);
    //m_pyramidImage.push_back(m_image);
#ifdef SFORESTS_EDGES
    m_sforestDetector = m_seq->getSFDetector();//cv::ximgproc::createStructuredEdgeDetection("../model/SForestModel.yml");
    m_pyramidImageSF.resize(EdgeVO::Settings::PYRAMID_DEPTH);
    m_pyramidImageSF[0] = cv::imread(m_imageName,cv::ImreadModes::IMREAD_COLOR);
#else
    m_sforestDetector = nullptr;
#endif
    
}

Frame::Frame(Mat& image, Mat& depthMap)
    :m_image(image) , m_depthMap(depthMap)
{}

/*
Frame::Frame(const Frame& cp)
: m_image( cp.m_image ) , m_depthMap( cp.m_depthMap )
{}
*/
Frame::~Frame()
{
    releaseAllVectors();
}

void Frame::releaseAllVectors()
{
    m_pyramidImage.clear();
    m_pyramid_Idx.clear();
    m_pyramid_Idy.clear();
    m_pyramidDepth.clear();
    m_pyramidMask.clear();
    m_pyramidEdge.clear();
    m_pyramidImageUINT.clear();

    m_pyramidEdge_binary.clear();
    m_pyramidDT_dists.clear();
    m_pyramidDT_labels.clear();

    //m_pyramidImageFloat.clear();
}

int Frame::getHeight(int lvl) const
{
    return m_pyramidImage[lvl].rows;
}
int Frame::getWidth(int lvl) const
{
    return m_pyramidImage[lvl].cols;
}
void Frame::printPaths() 
{
    std::cout << m_imageName << std::endl;
    std::cout << m_depthName << std::endl;
}
/*
Frame& Frame::operator=(const Frame& rhs)
{
    if(this == &rhs)
        return *this;
    // copy and swap
    Frame temp(rhs);
    std::swap(*this, temp);
    return *this;
}
*/
Mat& Frame::getImageForDisplayOnly()
{
    return m_pyramidImageUINT[0];
}
Mat& Frame::getEdgeForDisplayOnly()
{
    return m_pyramidEdge[0];
}
Mat& Frame::getDepthForDisplayOnly()
{
    return m_pyramidDepth[0];
}

cv::Mat Frame::getGX(int lvl) const
{
    return m_pyramid_Idx[lvl];
}
cv::Mat Frame::getGY(int lvl) const
{
    return m_pyramid_Idy[lvl];
}

Mat Frame::getImage(int lvl) const
{
    return m_pyramidImage[lvl].clone();
}
cv::Mat Frame::getImageVector(int lvl) const
{
    return (m_pyramidImage[lvl].clone()).reshape(1, m_pyramidImage[lvl].rows * m_pyramidImage[lvl].cols);
}

Mat Frame::getDepthMap(int lvl) const
{
    return (m_pyramidDepth[lvl].clone()).reshape(1, m_pyramidDepth[lvl].rows * m_pyramidDepth[lvl].cols);
}

Mat Frame::getMask(int lvl) const
{
    return m_pyramidMask[lvl].clone();
}
Mat Frame::getEdges(int lvl) const
{
    return (m_pyramidEdge[lvl].clone()).reshape(1, m_pyramidEdge[lvl].rows * m_pyramidEdge[lvl].cols);
}
cv::Mat Frame::getGradientX(int lvl) const
{
    return (m_pyramid_Idx[lvl].clone()).reshape(1, m_pyramid_Idx[lvl].rows * m_pyramid_Idx[lvl].cols);
}
cv::Mat Frame::getGradientY(int lvl) const
{
    return (m_pyramid_Idy[lvl].clone()).reshape(1, m_pyramid_Idy[lvl].rows * m_pyramid_Idy[lvl].cols);
}

//> cchien3 add: get distance transform map
cv::Mat Frame::getDistanceTransformMap(int lvl) const
{
    return (m_pyramidDT_dists[lvl].clone()).reshape(1, m_pyramidDT_dists[lvl].rows * m_pyramidDT_dists[lvl].cols);
}

//> cchien3 add: get distance transform label map
cv::Mat Frame::getDistanceTransformLabelMap(int lvl) const
{
    //return (m_pyramidDT_labels[lvl].clone());
    //cv::Mat A(120, 160, CV_32SC1);
    //for (int i = 0; i < 120 ; i++) {
        //for (int j = 0; j < 160 ; j++) {
            //A.at<int>(i,j) = i*160+j;        
        //}
    //}
    //return (A.clone()).reshape(1, A.rows * A.cols); 
    return (m_pyramidDT_labels[lvl].clone()).reshape(1, m_pyramidDT_labels[lvl].rows * m_pyramidDT_labels[lvl].cols);
}

void Frame::makePyramids()
{
    createPyramid(m_pyramidImage[0], m_pyramidImage, EdgeVO::Settings::PYRAMID_DEPTH, cv::INTER_LINEAR);
    cv::buildPyramid(m_pyramidImageUINT[0], m_pyramidImageUINT, EdgeVO::Settings::PYRAMID_BUILD);
    

#ifdef CANNY_EDGES
    // Canny
    createCannyEdgePyramids();
#elif LoG_EDGES
    // LoG
    createLoGEdgePyramids();
#elif SFORESTS_EDGES
    createStructuredForestEdgePyramid();
#elif CONV_BASIN
    createBasinPyramids();
#else
    // Sobel
    createSobelEdgePyramids();
#endif
    createImageGradientPyramids();
    createPyramid(m_pyramidDepth[0], m_pyramidDepth, EdgeVO::Settings::PYRAMID_DEPTH, cv::INTER_CUBIC);
    

}

void Frame::createPyramid(cv::Mat& src, std::vector<cv::Mat>& dst, int pyramidSize, int interpolationFlag)
{
    dst.resize(pyramidSize);
    dst[0] = src;
    for(size_t i = 1; i < pyramidSize; ++i)
        cv::resize(dst[i-1], dst[i],cv::Size(0, 0), 0.5, 0.5, interpolationFlag);
    
    
}

void Frame::createImageGradientPyramids()
{
    int one(1);
    int zero(0);
    double scale = 0.5;

    calcGradientX(m_pyramidImage[0], m_pyramid_Idx[0]);
    calcGradientY(m_pyramidImage[0], m_pyramid_Idy[0]);
   
    createPyramid(m_pyramid_Idx[0], m_pyramid_Idx, EdgeVO::Settings::PYRAMID_DEPTH, cv::INTER_CUBIC);
    createPyramid(m_pyramid_Idy[0], m_pyramid_Idy, EdgeVO::Settings::PYRAMID_DEPTH, cv::INTER_CUBIC);
}

void Frame::calcGX(cv::Mat &src)
{
    m_GX.resize(src.rows, src.cols);
    for(int y = 0; y < src.rows; ++y)
    {
        for(int x = 0; x < src.cols; ++x)
        {
            if(x == 0)
                m_GX(y,x) = (src.at<float>(y,x+1) - src.at<float>(y,x));
            else if(x == src.cols-1)
                m_GX(y,x) = (src.at<float>(y,x) - src.at<float>(y,x-1));
            else
                m_GX(y,x) = (src.at<float>(y,x+1) - src.at<float>(y,x-1))*0.5;
        }
    }
}

void Frame::calcGradientX(cv::Mat& src, cv::Mat& dst)
{
    dst = cv::Mat(src.rows, src.cols, CV_32FC1, 0.f).clone();
    for(int y = 0; y < src.rows; ++y)
    {
        for(int x = 0; x < src.cols; ++x)
        {
            if(x == 0)
                dst.at<float>(y,x) = (src.at<float>(y,x+1) - src.at<float>(y,x));
            else if(x == src.cols-1)
                dst.at<float>(y,x) = (src.at<float>(y,x) - src.at<float>(y,x-1));
            else
                dst.at<float>(y,x) = (src.at<float>(y,x+1) - src.at<float>(y,x-1))*0.5;
        }
    }
}
void Frame::calcGradientY(cv::Mat& src, cv::Mat& dst)
{
    dst = cv::Mat(src.rows, src.cols, CV_32FC1, 0.f ).clone();
    for(int y = 0; y < src.rows; ++y)
    {
        for(int x = 0; x < src.cols; ++x)
        {
            if(y == 0)
                dst.at<float>(y,x) = (src.at<float>(y+1,x) - src.at<float>(y,x));
            else if(y == src.rows-1)
                dst.at<float>(y,x) = (src.at<float>(y,x) - src.at<float>(y-1,x));
            else
                dst.at<float>(y,x) = (src.at<float>(y+1,x) - src.at<float>(y-1,x))*0.5;
        }
    }
}

std::string Frame::identifyMatType(cv::Mat img) {
    //> cchien3: add type identification
    int type = img.type();
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');
    printf("Edge Matrix: %s %dx%d \n", r.c_str(), img.cols, img.rows );
    return r;
}

void Frame::createCannyEdgePyramids()
{
    for(size_t i = 0; i < m_pyramidImage.size(); ++i)
    {
        Mat img_thresh; //not used
        //cv::GaussianBlur( m_pyramidImageUINT[i], m_pyramidImageUINT[i], Size(3,3), EdgeVO::Settings::SIGMA);
  /// Canny detector
        float upperThreshold = cv::threshold(m_pyramidImageUINT[i], img_thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        float lowerThresh = EdgeVO::Settings::CANNY_RATIO * upperThreshold;
        Canny(m_pyramidImageUINT[i], m_pyramidEdge[i], lowerThresh, upperThreshold, 3, true);

        

        //> cchien3: clone the edge map and binarize to fit the need of distance transform
        m_pyramidEdge_binary[i] = m_pyramidEdge[i].clone();
        for (int r = 0; r < m_pyramidEdge[i].rows; r++) {
            for (int c = 0; c < m_pyramidEdge[i].cols; c++) {
                m_pyramidEdge_binary[i].at<uchar>(r,c) = ((int)(m_pyramidEdge[i].at<uchar>(r,c)) == 0) ? 1 : 0;
            }
        }

        std::cout << "build pyramid check point 2" << std::endl;

        //> cchien3: write to a file
        /*if (i == 0) {
            std::cout << "Writing data to a file ..." << std::endl;
            std::ofstream edgeImg_file;
            std::string writeFileDir = "/users/yzhen105/data/yzhen105/github_repo/GeometricError//";
            writeFileDir.append("edgeImgData.txt");
            edgeImg_file.open(writeFileDir);
            if ( !edgeImg_file.is_open() ) std::cout << "Could not open the edgeImg_file!" << std::endl;
            for (int r = 0; r < m_pyramidEdge[i].rows; r++) {
                for (int c = 0; c < m_pyramidEdge[i].cols; c++) {
                    edgeImg_file << ((int)(m_pyramidEdge[i].at<uchar>(r,c))) << "\t";
                }
                edgeImg_file << "\n";
            }
            edgeImg_file.close();
        }

        if (i == 0) {
            std::cout << "Writing data to a file ..." << std::endl;
            std::ofstream edgeImg_file;
            std::string writeFileDir = "/users/yzhen105/data/yzhen105/github_repo/GeometricError//";
            writeFileDir.append("binaryImgData.txt");
            edgeImg_file.open(writeFileDir);
            if ( !edgeImg_file.is_open() ) std::cout << "Could not open the edgeImg_file!" << std::endl;
            for (int r = 0; r < m_pyramidEdge_binary[i].rows; r++) {
                for (int c = 0; c < m_pyramidEdge_binary[i].cols; c++) {
                    edgeImg_file << ((int)(m_pyramidEdge_binary[i].at<uchar>(r,c))) << "\t";
                }
                edgeImg_file << "\n";
            }
            edgeImg_file.close();
        }*/


        //> cchien3 add: create a distance transform distance map and a distance transform label map
        distanceTransform(m_pyramidEdge_binary[i], m_pyramidDT_dists[i], m_pyramidDT_labels[i], cv::DIST_L2, cv::DIST_MASK_PRECISE, cv::DIST_LABEL_PIXEL);

        //> check the cv::Mat class types
        std::string DT_type = identifyMatType(m_pyramidDT_dists[i]);
        std::string label_type = identifyMatType(m_pyramidDT_labels[i]);

        if (i == m_pyramidImage.size()-1 ) {
            imwrite( "/users/yzhen105/data/yzhen105/github_repo/GeometricError/Edges.jpg", m_pyramidEdge[i] );
            imwrite( "/users/yzhen105/data/yzhen105/github_repo/GeometricError/binary.jpg", m_pyramidEdge_binary[i] );
            //imwrite( "/users/yzhen105/data/yzhen105/github_repo/GeometricError//imgDT_dists.jpg", m_pyramidDT_dists[i] );
            //imwrite( "/users/yzhen105/data/yzhen105/github_repo/GeometricError//imgDT_lables.jpg", m_pyramidDT_labels[i] );
        }

        //> cchien3 TODO: WRITE DT AND LABEL RESULTS TO A FILE AND SEE WHETHER THE RESULTS LOOK GOOD!!!!!!
        if (i == m_pyramidImage.size()-1 && flag == 0) {
            flag = 1;
            std::cout << "Writing data to a file ..." << std::endl;
            std::ofstream dists_Img_file, label_img_file;
            std::string writeFileDir_dists = "/users/yzhen105/data/yzhen105/github_repo/GeometricError/";
            writeFileDir_dists.append("DT_Data.txt");
            std::string writeFileDir_label = "/users/yzhen105/data/yzhen105/github_repo/GeometricError/";
            writeFileDir_label.append("Label_Data.txt");
            dists_Img_file.open(writeFileDir_dists);
            label_img_file.open(writeFileDir_label);
            if ( !dists_Img_file.is_open() || label_img_file.is_open()) std::cout << "Could not open the files!" << std::endl;
            for (int r = 0; r < m_pyramidDT_dists[i].rows; r++) {
                for (int c = 0; c < m_pyramidDT_dists[i].cols; c++) {
                    dists_Img_file << ((m_pyramidDT_dists[i].at<float>(r,c))) << "\t";
                    label_img_file << ((m_pyramidDT_labels[i].at<int>(r,c))) << "\t";
                }
                dists_Img_file << "\n";
                label_img_file << "\n";
            }
            dists_Img_file.close();
            label_img_file.close();
        }
    }
    //void Canny(InputArray image, OutputArray edges, float threshold1, float threshold2, int apertureSize=3, bool L2gradient=false )
}

void Frame::createLoGEdgePyramids()
{
    for(size_t i = 0; i < m_pyramidImage.size(); ++i)
    {
        Mat img_dest; 
        cv::GaussianBlur( m_pyramidImageUINT[i], img_dest, Size(3,3), 0, 0, cv::BORDER_DEFAULT );
        cv::Laplacian( img_dest, img_dest, CV_8UC1, 3, 1., 0, cv::BORDER_DEFAULT );
        cv::convertScaleAbs( img_dest, img_dest );
        cv::threshold(img_dest, m_pyramidEdge[i], 25, 255, cv::THRESH_BINARY);
    }

}
void Frame::createSobelEdgePyramids()
{
    for(size_t i = 0; i < m_pyramidImage.size(); ++i)
    {
        cv::Mat grad_x, grad_y;
        cv::Mat grad;
        /// x Gradient
        Sobel( m_pyramidImageUINT[i], grad_x, CV_16S, 1, 0, 3, 1., 0, cv::BORDER_DEFAULT );
        convertScaleAbs( grad_x, grad_x );
        /// y Gradient
        Sobel( m_pyramidImageUINT[i], grad_y, CV_16S, 0, 1, 3, 1., 0, cv::BORDER_DEFAULT );
        convertScaleAbs( grad_y, grad_y );
        addWeighted( grad_x, 0.5, grad_y, 0.5, 0, grad );
        double max;
        double min;
        cv::minMaxLoc(grad, &min, &max);
        cv::threshold(grad/max, m_pyramidEdge[i], 0.95, 255, cv::THRESH_BINARY);
    }

}
void Frame::createStructuredForestEdgePyramid()
{
    cv::buildPyramid(m_pyramidImageSF[0], m_pyramidImageSF, EdgeVO::Settings::PYRAMID_BUILD);
    for(size_t i = 0; i < m_pyramidImageSF.size(); ++i)
    {
        Mat image = m_pyramidImageSF[i].clone();
        image.convertTo(image, CV_32FC3, 1./255.0);
        cv::Mat edges(image.size(), image.type());
        m_sforestDetector->detectEdges(image, edges );
        cv::threshold(edges, m_pyramidEdge[i], 0.15, 255, cv::THRESH_BINARY);
     
    }
        

}
void Frame::createBasinPyramids()
{
    for(size_t i = 0; i < m_pyramidImage.size(); ++i)
    {
        Mat img_thresh; //not used
        cv::GaussianBlur( m_pyramidImageUINT[i], m_pyramidImageUINT[i], Size(3,3), EdgeVO::Settings::SIGMA);
  /// Canny detector
        float upperThreshold = cv::threshold(m_pyramidImageUINT[i], img_thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        float lowerThresh = EdgeVO::Settings::CANNY_RATIO * upperThreshold;
        Canny(m_pyramidImageUINT[i], m_pyramidEdge[i], lowerThresh, upperThreshold, 3, true);
    }
    m_pyramidEdge[m_pyramidEdge.size()-1] = cv::Mat::ones(m_pyramidImageUINT[m_pyramidEdge.size()-1].rows, m_pyramidImageUINT[m_pyramidEdge.size()-1].cols, CV_8UC1);
}

bool Frame::hasDepthMap()
{
    return !(m_depthMap.empty() );

}

void Frame::setDepthMap(Mat& depthMap)
{
    if(!hasDepthMap())
        m_depthMap = depthMap;
    // Otherwise do nothing
}

}