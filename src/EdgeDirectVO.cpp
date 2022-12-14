#include "EdgeDirectVO.h"
#include "CycleTimer.h"
#include <algorithm>
#include <utility>
#include <iostream>
#include <iomanip>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "Pose.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <stdio.h> 
#include <stdlib.h> 
#include <random>
#include <iterator>
#include <algorithm>
#include <cmath>


namespace EdgeVO{
    using namespace cv;
EdgeDirectVO::EdgeDirectVO()
    :m_sequence(EdgeVO::Settings::ASSOC_FILE) , m_trajectory() , 
     m_lambda(0.)
{
    int length = m_sequence.getFrameHeight( getBottomPyramidLevel() ) * m_sequence.getFrameWidth( getBottomPyramidLevel() );
    
    m_X3DVector.resize(EdgeVO::Settings::PYRAMID_DEPTH); // Vector for each pyramid level
    for(size_t i = 0; i < m_X3DVector.size(); ++i)
        m_X3DVector[i].resize(length / std::pow(4, i) , Eigen::NoChange); //3 Vector for each pyramid for each image pixel

    //> cchien3: initialize m_X2D_ref_Vector
    m_X2D_ref_Vector.resize(EdgeVO::Settings::PYRAMID_DEPTH); // Vector for each pyramid level
    for(size_t i = 0; i < m_X2D_ref_Vector.size(); ++i)
        m_X2D_ref_Vector[i].resize(length / std::pow(4, i) , Eigen::NoChange); //2 Vector for each pyramid for each image pixel
    
    m_X2D_ref.resize(length, Eigen::NoChange);

    m_X3D.resize(length, Eigen::NoChange);
    m_warpedX.resize(length);
    m_warpedY.resize(length);
    m_refX.resize(length);
    m_refY.resize(length);
    m_warpedZ.resize(length);
    m_gx.resize(length);
    m_gxFinal.resize(length);
    m_gy.resize(length);
    m_gyFinal.resize(length);
    m_im1.resize(length);
    m_im1Final.resize(length);
    m_im2Final.resize(length);
    m_ZFinal.resize(length);
    m_Z.resize(length);
    m_edgeMask.resize(length);
    m_edgeMask_reference.resize(length);
    m_distLabel_reference.resize(length);

    m_outputFile.open(EdgeVO::Settings::RESULTS_FILE);

}

EdgeDirectVO::EdgeDirectVO(const EdgeDirectVO& cp)
    :m_sequence(EdgeVO::Settings::ASSOC_FILE)
{}

EdgeDirectVO::~EdgeDirectVO()
{
    m_outputFile.close();
}

EdgeDirectVO& EdgeDirectVO::operator=(const EdgeDirectVO& rhs)
{
    if(this == &rhs)
        return *this;
    
    EdgeDirectVO temp(rhs);
    std::swap(*this, temp);
    return *this;

}

void EdgeDirectVO::runEdgeDirectVO()
{
    //Start timer for stats
    m_statistics.start();
    //Make Pyramid for Reference frame
    m_sequence.makeReferenceFramePyramids();
    // Run for entire sequence

    //Prepare some vectors
    prepare3DPoints();

    //> cchien3: also prepare some vectors
    prepareEdgeCoordinates();

    //Init camera_pose with ground truth trajectory to make comparison easy
    Pose camera_pose = m_trajectory.initializePoseToGroundTruth(m_sequence.getFirstTimeStamp());
    Pose keyframe_pose = camera_pose;
    // relative_pose intiialized to identity matrix
    Pose relative_pose;

    // Start clock timer
    
    outputPose(camera_pose, m_sequence.getFirstTimeStamp());
    m_statistics.addStartTime((float) EdgeVO::CycleTimer::currentSeconds());

    //> cchien3: DEBUGGING!!
    //for (size_t n = 0; m_sequence.sequenceNotFinished(); ++n)
    for (size_t n = 0; n < 1; ++n)
    {
        std::cout << std::endl << camera_pose << std::endl;

//* yzhen105: DISPLAY_SEQUENCE not defined in Settings.h jump to else
//* yzhen105: to display, uncomment the line 13 in Settings.h
#ifdef DISPLAY_SEQUENCE
        //We re-use current frame for reference frame info
        m_sequence.makeCurrentFramePyramids();

        //Display images
        int keyPressed1 = m_sequence.displayCurrentImage();
        int keyPressed2 = m_sequence.displayCurrentEdge();
        int keyPressed3 = m_sequence.displayCurrentDepth();
        if(keyPressed1 == EdgeVO::Settings::TERMINATE_DISPLAY_KEY 
            || keyPressed2 == EdgeVO::Settings::TERMINATE_DISPLAY_KEY
            || keyPressed3 == EdgeVO::Settings::TERMINATE_DISPLAY_KEY) 
        {
            terminationRequested();
            break;
        }
        //Start algorithm timer for each iteration
        float startTime = (float) EdgeVO::CycleTimer::currentSeconds();
#else
        //Start algorithm timer for each iteration
        float startTime = (float) EdgeVO::CycleTimer::currentSeconds();
        m_sequence.makeCurrentFramePyramids();
#endif //DISPLAY_SEQUENCE

        if( n % EdgeVO::Settings::KEYFRAME_INTERVAL == 0 )
        {
            keyframe_pose = camera_pose;
            relative_pose.setIdentityPose();
        }
        //Constant motion assumption
        relative_pose.updateKeyFramePose(relative_pose.getPoseMatrix(), m_trajectory.getLastRelativePose());
        relative_pose.setPose(se3ExpEigen(se3LogEigen(relative_pose.getPoseMatrix())));

        //Constant acc. assumption
        //relative_pose.updateKeyFramePose(relative_pose.getPoseMatrix(), m_trajectory.get2LastRelativePose());
        //relative_pose.setPose(se3ExpEigen(se3LogEigen(relative_pose.getPoseMatrix())));
        
        // For each image pyramid level, starting at the top, going down
        //for (int lvl = getTopPyramidLevel(); lvl >= getBottomPyramidLevel(); --lvl)
        //{
        for (int lvl = getTopPyramidLevel(); lvl >= getTopPyramidLevel(); --lvl)
        {
            
            const Mat cameraMatrix(m_sequence.getCameraMatrix(lvl));
            prepareVectors(lvl);
            
            //make3DPoints(cameraMatrix, lvl);

            float lambda = 0.f;
            float error_last = EdgeVO::Settings::INF_F;
            float error = error_last;
            float error_geometry = error_last;
            for(int i = 0; i < EdgeVO::Settings::MAX_ITERATIONS_PER_PYRAMID[ lvl ]; ++i)
            {
                error_last = error;
                //* yzhen105:  + GeometricError(relative_pose.inversePoseEigen(), lvl)
                error = warpAndProject(relative_pose.inversePoseEigen(), lvl);// + GeometricError(relative_pose.inversePoseEigen(), lvl);

                //> cchien3: add geometrical error here
                std::cout << "Computing geometric error... " << std::endl;
                error_geometry = GeometricErrorFromCH(relative_pose.inversePoseEigen(), lvl);

                // Levenberg-Marquardt
                if( error < error_last)
                {
                    // Update relative pose
                    Eigen::Matrix<double, 6 , Eigen::RowMajor> del;
                    solveSystemOfEquations(lambda, lvl, del);
                    //std::cout << del << std::endl;

                    
                    if( (del.segment<3>(0)).dot(del.segment<3>(0)) < EdgeVO::Settings::MIN_TRANSLATION_UPDATE & 
                        (del.segment<3>(3)).dot(del.segment<3>(3)) < EdgeVO::Settings::MIN_ROTATION_UPDATE    )
                        break;

                    cv::Mat delMat = se3ExpEigen(del);
                    relative_pose.updatePose( delMat );

                    //Update lambda
                    if(lambda <= EdgeVO::Settings::LAMBDA_MAX)
                        lambda = EdgeVO::Settings::LAMBDA_MIN;
                    else
                        lambda *= EdgeVO::Settings::LAMBDA_UPDATE_FACTOR;
                }
                else
                {
                    if(lambda == EdgeVO::Settings::LAMBDA_MIN)
                        lambda = EdgeVO::Settings::LAMBDA_MAX;
                    else
                        lambda *= EdgeVO::Settings::LAMBDA_UPDATE_FACTOR;
                }
            }
        }
        camera_pose.updateKeyFramePose(keyframe_pose.getPoseMatrix(), relative_pose.getPoseMatrix());
        outputPose(camera_pose, m_sequence.getCurrentTimeStamp());
        //At end, update sequence for next image pair
        float endTime = (float) EdgeVO::CycleTimer::currentSeconds();
        m_trajectory.addPose(camera_pose);
                
        // Don't time past this part (reading from disk)
        m_sequence.advanceSequence();
        m_statistics.addDurationForFrame(startTime, endTime);
        m_statistics.addCurrentTime((float) EdgeVO::CycleTimer::currentSeconds());
        m_statistics.printStatistics();
        
    }
    // End algorithm level timer
    m_statistics.end();
    return;
}

void EdgeDirectVO::prepareVectors(int lvl)
{
    cv2eigen(m_sequence.getReferenceFrame()->getDepthMap(lvl), m_Z);
    cv2eigen(m_sequence.getCurrentFrame()->getEdges(lvl), m_edgeMask); //* yzhen105: canny
    cv2eigen(m_sequence.getReferenceFrame()->getImageVector(lvl), m_im1);
    cv2eigen(m_sequence.getCurrentFrame()->getImageVector(lvl), m_im2);
    cv2eigen(m_sequence.getCurrentFrame()->getGradientX(lvl), m_gx);
    cv2eigen(m_sequence.getCurrentFrame()->getGradientY(lvl), m_gy);
    
    //* yzhen105: canny edge for the reference image
    cv2eigen(m_sequence.getReferenceFrame()->getEdges(lvl), m_edgeMask_reference); 
    //> cchien3: get distance transform label map for the reference image
    cv2eigen(m_sequence.getReferenceFrame()->getDistanceTransformLabelMap(lvl), m_distLabel_reference); 
    //m_distLabel_reference.resize(160*120,1);
    
    std::cout << "Size of m_distLabel_reference: " << m_distLabel_reference.rows() << std::endl;    
    
    size_t numElements;
////////////////////////////////////////////////////////////
// REGULAR_DIRECT_VO
////////////////////////////////////////////////////////////
#ifdef REGULAR_DIRECT_VO
    m_edgeMask = (m_edgeMask.array() == 0).select(1, m_edgeMask);
    m_edgeMask = (m_Z.array() <= 0.f).select(0, m_edgeMask);
    m_edgeMask = (m_gx.array()*m_gx.array() + m_gy.array()*m_gy.array() <= EdgeVO::Settings::MIN_GRADIENT_THRESH).select(0, m_edgeMask);
#elif REGULAR_DIRECT_VO_SUBSET
    m_edgeMask = (m_Z.array() <= 0.f).select(0, m_edgeMask);
    numElements = (m_edgeMask.array() != 0).count() * EdgeVO::Settings::PERCENT_EDGES;
    m_edgeMask = (m_edgeMask.array() == 0).select(1, m_edgeMask);
    m_edgeMask = (m_Z.array() <= 0.f).select(0, m_edgeMask);

#else
    m_edgeMask = (m_Z.array() <= 0.f).select(0, m_edgeMask);
    //* yzhen105: For reference image
    //m_edgeMask_reference = (m_Z.array() <= 0.f).select(0, m_edgeMask_reference);
    
    //m_edgeMask = (m_Z.array() <= 0.f).select(0, m_edgeMask);
    //size_t numElements = (m_edgeMask.array() != 0).count() * EdgeVO::Settings::PERCENT_EDGES;
#endif //REGULAR_DIRECT_VO

////////////////////////////////////////////////////////////
// EDGEVO_SUBSET_POINTS
////////////////////////////////////////////////////////////
#ifdef EDGEVO_SUBSET_POINTS
    //numElements = (m_edgeMask.array() != 0).count() * EdgeVO::Settings::PERCENT_EDGES;
    //size_t numElements = (m_edgeMask.array() != 0).count() < EdgeVO::Settings::NUMBER_POINTS ? (m_edgeMask.array() != 0).count() : EdgeVO::Settings::NUMBER_POINTS;
    std::vector<size_t> indices, randSample;
    m_im1Final.resize(numElements);
    m_XFinal.resize(numElements);
    m_YFinal.resize(numElements);
    m_ZFinal.resize(numElements);
    m_X3D.resize(numElements ,Eigen::NoChange);
    m_finalMask.resize(numElements);

    //size_t idx = 0;
    for(int i = 0; i < m_edgeMask.rows(); ++i)
    {
        if(m_edgeMask[i] != 0)
        {
            indices.push_back(i);
        }
    }
    std::sample(indices.begin(), indices.end(), std::back_inserter(randSample),
                numElements, std::mt19937{std::random_device{}()});
    
    //size_t idx = 0;
    for(int i = 0; i < randSample.size(); ++i)
    {
        m_im1Final[i] = m_im1[randSample[i]];
        m_ZFinal[i] = m_Z[randSample[i]];
        m_X3D.row(i) = (m_X3DVector[lvl].row(randSample[i])).array() * m_Z[randSample[i]];
        m_finalMask[i] = m_edgeMask[randSample[i]];    
    }


#else
////////////////////////////////////////////////////////////
// Edge Direct VO
////////////////////////////////////////////////////////////

    //> cchien3 add
    numElements = (m_edgeMask.array() != 0).count();
    
    m_im1Final.resize(numElements);
    m_XFinal.resize(numElements);
    m_YFinal.resize(numElements);
    m_ZFinal.resize(numElements);
    m_X3D.resize(numElements ,Eigen::NoChange);
    m_finalMask.resize(numElements);

    size_t idx = 0;
    for(int i = 0; i < m_edgeMask.rows(); ++i)
    {
        if(m_edgeMask[i] != 0)
        {
            m_im1Final[idx] = m_im1[i];
            m_ZFinal[idx] = m_Z[i];
            m_X3D.row(idx) = (m_X3DVector[lvl].row(i)).array() * m_Z[i];
            m_finalMask[idx] = m_edgeMask[i];
            ++idx;
        }
    }

    //> cchien3 ========================================================================
	size_t numElements_ref = (m_edgeMask_reference.array() != 0).count();
	size_t idx_ref = 0;

    //* yzhen105: get the ccoordinates for reference edge pixels
	//int idx_vec = 0;
    const int w = m_sequence.getFrameWidth(lvl);
    const int h = m_sequence.getFrameHeight(lvl);
    m_finalMask_reference.resize(numElements_ref);

    std::cout << "Size of m_edgeMask_reference: " << m_edgeMask_reference.rows() << std::endl;
    std::cout << "Size of nonzero m_edgeMask_reference: " << numElements_ref << std::endl;

    //> cchien3: fix the Segmentation fault issue
    //> Now that m_X2D_ref_Vector[lvl].row(i) is a std::vector storing ALL pixel coordinates (x,y) by (m_X2D_ref_Vector[lvl](i,0), m_X2D_ref_Vector[lvl](i,1))
    //> if you want the whole row, simply use m_X2D_ref_Vector[lvl].row(i)
    //> m_X2D_ref is a Eigen matrix of Nx2 storing edge-only pixel coordinates (x,y)
    //> 
    int test_i = 0;
    for(int i = 0; i < m_edgeMask_reference.rows(); ++i)
    {
        //> cchien3: if it is an edge pixel
        if(m_edgeMask_reference[i] != 0)
        {
            if (idx_ref == 0) {
                test_i = i;
                std::cout << i << std::endl;
            }
            m_finalMask_reference[idx_ref] = m_edgeMask_reference[i];
            m_X2D_ref.row(idx_ref) = (m_X2D_ref_Vector[lvl].row(i)).array();
            ++idx_ref;
        }
    }

    std::cout << "Size of m_X2D_ref: " << m_X2D_ref.rows() << std::endl;
    std::cout << idx_ref << std::endl;

    //> TEST TO UNDERSTAND HOW EIGEN WORKS ...
    std::cout << "level " << lvl << ":" << std::endl;
    std::cout << m_X2D_ref.row(0) << std::endl;
    std::cout << m_X2D_ref.row(1) << std::endl;
    std::cout << m_X2D_ref.row(0) - m_X2D_ref.row(1) << std::endl;
    std::cout << m_X2D_ref(0,1) << std::endl;
    std::cout << (m_X2D_ref_Vector[lvl].row(test_i)).array() << std::endl;

    //> cchien3: write all edge pixels of the bottom pyramid image to a file for validations
    if (lvl == getBottomPyramidLevel()) {
        std::cout << "Writing data to a file ..." << std::endl;
        std::ofstream edgeCoordinates_file;
        std::string writeFileDir = "/users/yzhen105/data/yzhen105/github_repo/GeometricError/";
        writeFileDir.append("refImg_edgeCoordinates.txt");
        edgeCoordinates_file.open(writeFileDir);
        if ( !edgeCoordinates_file.is_open() ) {
            std::cout << "Could not open the edgeCoordinates_file!" << std::endl;
        }

        for (int i = 0; i < m_finalMask_reference.rows(); i++) {
            edgeCoordinates_file << m_X2D_ref(i,0) << "\t" << m_X2D_ref(i,1) << "\n";
        }

        edgeCoordinates_file.close();
    }
    m_edgeMask_reference.resize(numElements_ref);
	m_edgeMask_reference = m_finalMask_reference;

#endif //EDGEVO_SUBSET_POINTS
////////////////////////////////////////////////////////////
    m_Z.resize(numElements);
    m_Z = m_ZFinal;
    m_edgeMask.resize(numElements);
    m_edgeMask = m_finalMask;

    std::cout << "prepare Vectors check point 4" << std::endl;
    
}

void EdgeDirectVO::make3DPoints(const cv::Mat& cameraMatrix, int lvl)
{
    m_X3D = m_X3DVector[lvl].array() * m_Z.replicate(1, m_X3DVector[lvl].cols() ).array();
}

float EdgeDirectVO::warpAndProject(const Eigen::Matrix<double,4,4>& invPose, int lvl)
{
    Eigen::Matrix<float,3,3> R = (invPose.block<3,3>(0,0)).cast<float>() ;
    Eigen::Matrix<float,3,1> t = (invPose.block<3,1>(0,3)).cast<float>() ;
    //std::cout << R << std::endl << t << std::endl;
    //std::cout << "Cols: " << m_X3D[lvl].cols() << "Rows: " << m_X3D[lvl].rows() << std::endl;
    
    m_newX3D.resize(Eigen::NoChange, m_X3D.rows());
    m_newX3D = R * m_X3D.transpose() + t.replicate(1, m_X3D.rows() );

    const Mat cameraMatrix(m_sequence.getCameraMatrix(lvl));
    const float fx = cameraMatrix.at<float>(0, 0);
    const float cx = cameraMatrix.at<float>(0, 2);
    const float fy = cameraMatrix.at<float>(1, 1);
    const float cy = cameraMatrix.at<float>(1, 2);
    //std::cout << cy << std::endl;
    //exit(1);
    const int w = m_sequence.getFrameWidth(lvl);
    const int h = m_sequence.getFrameHeight(lvl);

    m_warpedX.resize(m_X3D.rows());
    m_warpedY.resize(m_X3D.rows());

    m_warpedX = (fx * (m_newX3D.row(0)).array() / (m_newX3D.row(2)).array() ) + cx;
    //m_warpedX.array() += cx;
    m_warpedY = (fy * (m_newX3D.row(1)).array() / (m_newX3D.row(2)).array() ) + cy;
    //m_warpedY.array() += cy;

    // (R.array() < s).select(P,Q );  // (R < s ? P : Q)
    //std::cout << newX3D.rows() << std::endl;
    //std::cout << m_finalMask.rows() << std::endl;

    // Check both Z 3D points are >0
    //m_finalMask = m_edgeMask;

    m_finalMask = m_edgeMask;

    m_finalMask = (m_newX3D.row(2).transpose().array() <= 0.f).select(0, m_finalMask);
    //m_finalMask = (m_newX3D.row(2).transpose().array() > EdgeVO::Settings::MAX_Z_DEPTH).select(0, m_finalMask);

    //m_finalMask = (m_newX3D.row(2).transpose().array() > 10.f).select(0, m_finalMask);
    m_finalMask = (m_X3D.col(2).array() <= 0.f).select(0, m_finalMask);
    //m_finalMask = (m_X3D.col(2).array() > 10.f).select(0, m_finalMask);
    m_finalMask = ( (m_X3D.col(2).array()).isFinite() ).select(m_finalMask, 0);
    m_finalMask = ( (m_newX3D.row(2).transpose().array()).isFinite() ).select(m_finalMask, 0);
    
    // Check new projected x coordinates are: 0 <= x < w-1
    m_finalMask = (m_warpedX.array() < 0.f).select(0, m_finalMask);
    m_finalMask = (m_warpedX.array() >= w-2).select(0, m_finalMask);
    m_finalMask = (m_warpedX.array().isFinite()).select(m_finalMask, 0);
    // Check new projected x coordinates are: 0 <= y < h-1
    m_finalMask = (m_warpedY.array() >= h-2).select(0, m_finalMask);
    m_finalMask = (m_warpedY.array() < 0.f).select(0, m_finalMask);
    m_finalMask = (m_warpedY.array().isFinite()).select(m_finalMask, 0);
    

// If we want every point, save some computation time- see the #else
////////////////////////////////////////////////////////////
#ifdef EDGEVO_SUBSET_POINTS_EXACT
    size_t numElements = (m_finalMask.array() != 0).count() < EdgeVO::Settings::NUMBER_POINTS ? (m_finalMask.array() != 0).count() : EdgeVO::Settings::NUMBER_POINTS;

    //size_t numElements = (m_finalMask.array() != 0).count();
    m_gxFinal.resize(numElements);
    m_gyFinal.resize(numElements);
    m_im1.resize(numElements);
    m_im2Final.resize(numElements);
    m_XFinal.resize(numElements);
    m_YFinal.resize(numElements);
    m_ZFinal.resize(numElements);
    std::vector<size_t> indices, randSample;

    //size_t idx = 0;
    for(int i = 0; i < m_finalMask.rows(); ++i)
    {
        if(m_finalMask[i] != 0)
        {
            indices.push_back(i);
        }
    }
    std::sample(indices.begin(), indices.end(), std::back_inserter(randSample),
                numElements, std::mt19937{std::random_device{}()});
    
    size_t idx = 0;
    for(int i = 0; i < randSample.size(); ++i)
    {
        m_gxFinal[i]  = interpolateVector( m_gx, m_warpedX[randSample[i]], m_warpedY[randSample[i]], w);
        m_gyFinal[i]  = interpolateVector( m_gy, m_warpedX[randSample[i]], m_warpedY[randSample[i]], w);
        m_im1[i] = m_im1Final[randSample[i]];//interpolateVector(m_im1, m_warpedX[i], m_warpedY[i], w);
        m_im2Final[i] = interpolateVector(m_im2, m_warpedX[randSample[i]], m_warpedY[randSample[i]], w);
        m_XFinal[i] = m_newX3D(0,randSample[i]);
        m_YFinal[i] = m_newX3D(1,randSample[i]);
        m_ZFinal[i] = m_newX3D(2,randSample[i]);        
    }
    
////////////////////////////////////////////////////////////
#else //EDGEVO_SUBSET_POINTS_EXACT
    // For non random numbers EDGEVO_SUBSET_POINTS
    size_t numElements = (m_finalMask.array() != 0).count();
    m_gxFinal.resize(numElements);
    m_gyFinal.resize(numElements);
    m_im1.resize(numElements);
    m_im2Final.resize(numElements);
    m_XFinal.resize(numElements);
    m_YFinal.resize(numElements);
    m_ZFinal.resize(numElements);

    size_t idx = 0;
    for(int i = 0; i < m_finalMask.rows(); ++i)
    {
        if(m_finalMask[i] != 0)
        {
            m_gxFinal[idx]  = interpolateVector( m_gx, m_warpedX[i], m_warpedY[i], w);
            m_gyFinal[idx]  = interpolateVector( m_gy, m_warpedX[i], m_warpedY[i], w);
            m_im1[idx] = m_im1Final[i];//interpolateVector(m_im1, m_warpedX[i], m_warpedY[i], w);
            m_im2Final[idx] = interpolateVector(m_im2, m_warpedX[i], m_warpedY[i], w);
            m_XFinal[idx] = m_newX3D(0,i);
            m_YFinal[idx] = m_newX3D(1,i);
            m_ZFinal[idx] = m_newX3D(2,i);
            
            ++idx;
        }
    }
#endif //EDGEVO_SUBSET_POINTS_EXACT
////////////////////////////////////////////////////////////
    
    //apply mask to im1, im2, gx, and gy
    //interp coordinates of im2, gx, and gy
    // calc residual

    //calc A and b matrices
    //
    m_residual.resize(numElements);
    m_rsquared.resize(numElements);
    m_weights.resize(numElements);

    m_residual = ( m_im1.array() - m_im2Final.array() );
    m_rsquared = m_residual.array() * m_residual.array();

    m_weights = Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor>::Ones(numElements);
    m_weights = ( ( (m_residual.array()).abs() ) > EdgeVO::Settings::HUBER_THRESH ).select( EdgeVO::Settings::HUBER_THRESH / (m_residual.array()).abs() , m_weights);

    return ( (m_weights.array() * m_rsquared.array()).sum() / (float) numElements );
     
}

void EdgeDirectVO::solveSystemOfEquations(const float lambda, const int lvl, Eigen::Matrix<double, 6 , Eigen::RowMajor>& poseupdate)
{
    const Mat cameraMatrix(m_sequence.getCameraMatrix(lvl));
    const float fx = cameraMatrix.at<float>(0, 0);
    const float cx = cameraMatrix.at<float>(0, 2);
    const float fy = cameraMatrix.at<float>(1, 1);
    const float cy = cameraMatrix.at<float>(1, 2);

    size_t numElements = m_im2Final.rows();
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> Z2 = m_ZFinal.array() * m_ZFinal.array();

    m_Jacobian.resize(numElements, Eigen::NoChange);
    m_Jacobian.col(0) =  m_weights.array() * fx * ( m_gxFinal.array() / m_ZFinal.array() );

    m_Jacobian.col(1) =  m_weights.array() * fy * ( m_gyFinal.array() / m_ZFinal.array() );

    m_Jacobian.col(2) = - m_weights.array()* ( fx * ( m_XFinal.array() * m_gxFinal.array() ) + fy * ( m_YFinal.array() * m_gyFinal.array() ) )
                        / ( Z2.array() );

    m_Jacobian.col(3) = - m_weights.array() * ( fx * m_XFinal.array() * m_YFinal.array() * m_gxFinal.array() / Z2.array()
                         + fy *( 1.f + ( m_YFinal.array() * m_YFinal.array() / Z2.array() ) ) * m_gyFinal.array() );

    m_Jacobian.col(4) = m_weights.array() * ( fx * (1.f + ( m_XFinal.array() * m_XFinal.array() / Z2.array() ) ) * m_gxFinal.array() 
                        + fy * ( m_XFinal.array() * m_YFinal.array() * m_gyFinal.array() ) / Z2.array() );

    m_Jacobian.col(5) = m_weights.array() * ( -fx * ( m_YFinal.array() * m_gxFinal.array() ) + fy * ( m_XFinal.array() * m_gyFinal.array() ) )
                        / m_ZFinal.array();
    
    m_residual.array() *= m_weights.array();
    
    poseupdate = -( (m_Jacobian.transpose() * m_Jacobian).cast<double>() ).ldlt().solve( (m_Jacobian.transpose() * m_residual).cast<double>() );

    
}
float EdgeDirectVO::interpolateVector(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor>& toInterp, float x, float y, int w) const
{
    int xi = (int) x;
	int yi = (int) y;
	float dx = x - xi;
	float dy = y - yi;
	float dxdy = dx * dy;
    int topLeft = w * yi + xi;
    int topRight = topLeft + 1;
    int bottomLeft = topLeft + w;
    int bottomRight= bottomLeft + 1;
  
    //               x                x+1
    //       ======================================
    //  y    |    topLeft      |    topRight      |
    //       ======================================
    //  y+w  |    bottomLeft   |    bottomRight   |
    //       ======================================
    return  dxdy * toInterp[bottomRight]
	        + (dy - dxdy) * toInterp[bottomLeft]
	        + (dx - dxdy) * toInterp[topRight]
			+ (1.f - dx - dy + dxdy) * toInterp[topLeft];
}
void EdgeDirectVO::prepare3DPoints( )
{
    
    for (int lvl = 0; lvl < EdgeVO::Settings::PYRAMID_DEPTH; ++lvl)
    {
        const Mat cameraMatrix(m_sequence.getCameraMatrix(lvl));
        int w = m_sequence.getFrameWidth(lvl);
        int h = m_sequence.getFrameHeight(lvl);
        const float fx = cameraMatrix.at<float>(0, 0);
        const float cx = cameraMatrix.at<float>(0, 2);
        const float fy = cameraMatrix.at<float>(1, 1);
        const float cy = cameraMatrix.at<float>(1, 2);
        const float fxInv = 1.f / fx;
        const float fyInv = 1.f / fy;
    
        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                int idx = y * w + x;
                m_X3DVector[lvl].row(idx) << (x - cx) * fxInv, (y - cy) * fyInv, 1.f ;
            }
        }
    }
}

//> cchien3: assign pixel coordinates to a 2D image plane m_X2D_ref_Vector
void EdgeDirectVO::prepareEdgeCoordinates( )
{
    for (int lvl = 0; lvl < EdgeVO::Settings::PYRAMID_DEPTH; ++lvl)
    {
        int w = m_sequence.getFrameWidth(lvl);
        int h = m_sequence.getFrameHeight(lvl);

        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                int idx = y * w + x;
                m_X2D_ref_Vector[lvl].row(idx) << x, y;
            }
        }
    }    
}

void EdgeDirectVO::warpAndCalculateResiduals(const Pose& pose, const std::vector<float>& Z, const std::vector<bool>& E, const int h, const int w, const cv::Mat& cameraMatrix, const int lvl)
{
    const int ymax = h;
    const int xmax = w;
    const int length = xmax * ymax;

    const float fx = cameraMatrix.at<float>(0,0);
    const float cx = cameraMatrix.at<float>(0,2);
    const float fy = cameraMatrix.at<float>(1,1);
    const float cy = cameraMatrix.at<float>(1,2);

    const Mat inPose( m_trajectory.getCurrentPose().inversePose() );
    Eigen::Matrix<float,4,4> invPose;
    cv::cv2eigen(inPose,invPose);


    for(int i = 0; i < ymax*xmax; ++i)
    {
        float z3d = Z[i];
        float x = i / ymax;
        float y = i % xmax;
        float x3d = z3d * (x - cx)/ fx;
        float y3d = z3d * (y - cy)/ fy;
    }
    return;
}

inline
bool EdgeDirectVO::checkBounds(float x, float xlim, float y, float ylim, float oldZ, float newZ, bool edgePixel)
{
    return ( (edgePixel) & (x >= 0) & x < xlim & y >= 0 & y < ylim & oldZ >= 0. & newZ >= 0. );
        
}
void EdgeDirectVO::terminationRequested()
{
    printf("Display Terminated by User\n");
    m_statistics.printStatistics();

}

void EdgeDirectVO::outputPose(const Pose& pose, double timestamp)
{
    Eigen::Matrix<double,4,4,Eigen::RowMajor> T;
    cv::Mat pmat = pose.getPoseMatrix();
    cv::cv2eigen(pmat,T);
    Eigen::Matrix<double,3,3,Eigen::RowMajor> R = T.block<3,3>(0,0);
    Eigen::Matrix<double,3,Eigen::RowMajor> t = T.block<3,1>(0,3);
    Eigen::Quaternion<double> quat(R);

    m_outputFile << std::setprecision(EdgeVO::Settings::RESULTS_FILE_PRECISION) << std::fixed << std::showpoint << timestamp;
    m_outputFile << " ";
    m_outputFile << std::setprecision(EdgeVO::Settings::RESULTS_FILE_PRECISION) << std::fixed << std::showpoint << t[0];
    m_outputFile << " ";
    m_outputFile << std::setprecision(EdgeVO::Settings::RESULTS_FILE_PRECISION) << std::fixed << std::showpoint << t[1];
    m_outputFile << " ";
    m_outputFile << std::setprecision(EdgeVO::Settings::RESULTS_FILE_PRECISION) << std::fixed << std::showpoint << t[2];
    m_outputFile << " ";
    m_outputFile << std::setprecision(EdgeVO::Settings::RESULTS_FILE_PRECISION) << std::fixed << std::showpoint << quat.x();
    m_outputFile << " ";
    m_outputFile << std::setprecision(EdgeVO::Settings::RESULTS_FILE_PRECISION) << std::fixed << std::showpoint << quat.y();
    m_outputFile << " ";
    m_outputFile << std::setprecision(EdgeVO::Settings::RESULTS_FILE_PRECISION) << std::fixed << std::showpoint << quat.z();
    m_outputFile << " ";
    m_outputFile << std::setprecision(EdgeVO::Settings::RESULTS_FILE_PRECISION) << std::fixed << std::showpoint << quat.w();
    m_outputFile << std::endl;
}


//* yzhen105: Geometric Error Calculation
float EdgeDirectVO::GeometricError(const Eigen::Matrix<double,4,4>& invPose, int lvl)
{
    //* yzhen105: initialize the R and T matrix here
    Eigen::Matrix<float,3,3> R = (invPose.block<3,3>(0,0)).cast<float>() ;
    Eigen::Matrix<float,3,1> t = (invPose.block<3,1>(0,3)).cast<float>() ;
   
    //* yzhen105: resize 3D new image
    m_newX3D.resize(Eigen::NoChange, m_X3D.rows());
    //m_newX2D_ref.resize(Eigen::NoChange, m_X2D_ref.rows());
    //* yzhen105: reprojection
    m_newX3D     = R * m_X3D.transpose() + t.replicate(1, m_X3D.rows() );
    //m_newX2D_ref = m_X2D_ref;

    //* yzhen105: get the camera matrix
    const Mat cameraMatrix(m_sequence.getCameraMatrix(lvl));
    const float fx = cameraMatrix.at<float>(0, 0);
    const float cx = cameraMatrix.at<float>(0, 2);
    const float fy = cameraMatrix.at<float>(1, 1);
    const float cy = cameraMatrix.at<float>(1, 2);
    //* yzhen105: get the size of current image
    const int w = m_sequence.getFrameWidth(lvl);
    const int h = m_sequence.getFrameHeight(lvl);

   
    //* yzhen105: get the reprojected coordinates of edge pixels in new image
    m_warpedX = (fx * (m_newX3D.row(0)).array() / (m_newX3D.row(2)).array() ) + cx;
    //m_refX    = m_newX2D_ref.row(0).array();
    //m_warpedX.array() += cx;
    m_warpedY = (fy * (m_newX3D.row(1)).array() / (m_newX3D.row(2)).array() ) + cy;
    //m_refY    = m_newX2D_ref.row(1).array();

    m_finalMask = m_edgeMask;

    m_finalMask = (m_newX3D.row(2).transpose().array() <= 0.f).select(0, m_finalMask);
    m_finalMask = (m_X3D.col(2).array() <= 0.f).select(0, m_finalMask);
    m_finalMask = ( (m_X3D.col(2).array()).isFinite() ).select(m_finalMask, 0);
    m_finalMask = ( (m_newX3D.row(2).transpose().array()).isFinite() ).select(m_finalMask, 0);
   
    // Check new projected x coordinates are: 0 <= x < w-1
    m_finalMask = (m_warpedX.array() < 0.f).select(0, m_finalMask);
    m_finalMask = (m_warpedX.array() >= w-2).select(0, m_finalMask);
    m_finalMask = (m_warpedX.array().isFinite()).select(m_finalMask, 0);
    // Check new projected x coordinates are: 0 <= y < h-1
    m_finalMask = (m_warpedY.array() >= h-2).select(0, m_finalMask);
    m_finalMask = (m_warpedY.array() < 0.f).select(0, m_finalMask);
    m_finalMask = (m_warpedY.array().isFinite()).select(m_finalMask, 0);

    size_t idx_cor = 0;
    size_t idx = 0;
    size_t numElements = (m_finalMask.array() != 0).count();
    float prev_distance;
    //size_t numElements = (m_warpedX.array() != 0).count();
    m_residual_GE.resize(2*numElements);

    euc_distance_sum = 0.f;

    //std::ofstream edgeCoordinates_file;
    //std::string writeFileDir = "/users/yzhen105/data/yzhen105/EdgeDirectVO/";
    //writeFileDir.append("coord_results.txt");
    //edgeCoordinates_file.open(writeFileDir);
    //std::cout << "numElements is " << numElements << std::endl;

    std::cout << "========================================================" << std::endl;
    std::cout << numElements << std::endl;
    std::cout << m_warpedX.rows() << std::endl;

    for(int i = 0; i <  m_warpedX.rows(); ++i)    // new img coordinate
    {
        if(m_finalMask[i] != 0)
        {
            prev_distance = EdgeVO::Settings::INF_F;
            //std::cout << "2. run here " << std::endl;
            //std::cout << "m_X2D_ref(5,0) " << m_X2D_ref(5,0) << std::endl;
            //std::cout << "m_X2D_ref(5,1) " << m_X2D_ref(5,1) << std::endl;
            //std::cout << "m_warpedX[2] " << m_warpedX[2] << std::endl;
            for(int j = 0; j <  m_X2D_ref.rows(); ++j)  // ref img coordinate
            {
                //std::cout << "3. run here " << std::endl;
                //* yzhen105: calculate the euclidean distance
                //distance = sqrt(pow((m_warpedX[i] - m_refX[j]),2)+pow((m_warpedY[i] - m_refX[j]),2)); //* yzhen105: #include <cmath>
                distance = (m_warpedX[i] - m_X2D_ref(j,0))*(m_warpedX[i] - m_X2D_ref(j,0))+(m_warpedY[i] - m_X2D_ref(j,1))*(m_warpedY[i] - m_X2D_ref(j,1)); //* yzhen105: #include <cmath>
                //std::cout << "4. run here " << std::endl;
                //std::cout << "distance is " << std::endl;
                //std::cout << std::endl << distance << std::endl;
                if(distance < prev_distance)
                {
                    prev_distance = distance;
                    idx_cor       = j;
                    //std::cout << "new distance is " << prev_distance << std::endl;
                    //std::cout << "with i " << i << ", j " << j << ", idx " << idx << std::endl;
                    //std::cout << "m_residual_GE[2*idx]: " << m_residual_GE[2*idx] << ", m_residual_GE[2*idx+1] " << m_residual_GE[2*idx+1] << std::endl;
                }
            }
            //edgeCoordinates_file << m_X2D_ref(idx_cor,0) << "\t" << m_X2D_ref(idx_cor,1) << "\n";
            m_residual_GE[2*idx]   = m_warpedX[i] - m_X2D_ref(idx_cor,0) ;
            m_residual_GE[2*idx+1] = m_warpedY[i] - m_X2D_ref(idx_cor,1) ;
            //std::cout << "m_warpedX.rows() is " <<  m_warpedX.rows() << ", m_X2D_ref.rows() is " << m_X2D_ref.rows() << std::endl;
            ++idx;
            euc_distance_sum += prev_distance;
            //std::cout << "5. run here " << std::endl;
        }
    }
    //edgeCoordinates_file.close();
    //std::cout << "6. run here " << std::endl;
    //* yzhen105: should return the sum of (x_bar-x)^2 ???error for geometric
    return (euc_distance_sum);
}

//* yzhen105: Geometric Error Calculation
float EdgeDirectVO::GeometricErrorFromCH(const Eigen::Matrix<double,4,4>& invPose, int lvl)
{
    //* yzhen105: initialize the R and T matrix here
    Eigen::Matrix<float,3,3> R = (invPose.block<3,3>(0,0)).cast<float>() ;
    Eigen::Matrix<float,3,1> t = (invPose.block<3,1>(0,3)).cast<float>() ;
   
    //* yzhen105: resize 3D new image
    m_newX3D.resize(Eigen::NoChange, m_X3D.rows());
    //m_newX2D_ref.resize(Eigen::NoChange, m_X2D_ref.rows());
    //* yzhen105: reprojection
    m_newX3D     = R * m_X3D.transpose() + t.replicate(1, m_X3D.rows() );

    //* yzhen105: get the camera matrix
    const Mat cameraMatrix(m_sequence.getCameraMatrix(lvl));
    const float fx = cameraMatrix.at<float>(0, 0);
    const float cx = cameraMatrix.at<float>(0, 2);
    const float fy = cameraMatrix.at<float>(1, 1);
    const float cy = cameraMatrix.at<float>(1, 2);
    //* yzhen105: get the size of current image
    const int w = m_sequence.getFrameWidth(lvl);
    const int h = m_sequence.getFrameHeight(lvl);

    //* yzhen105: get the reprojected coordinates of edge pixels in new image
    m_warpedX = (fx * (m_newX3D.row(0)).array() / (m_newX3D.row(2)).array() ) + cx;
    //m_warpedX.array() += cx;
    m_warpedY = (fy * (m_newX3D.row(1)).array() / (m_newX3D.row(2)).array() ) + cy;

    m_finalMask = m_edgeMask;

    m_finalMask = (m_newX3D.row(2).transpose().array() <= 0.f).select(0, m_finalMask);
    m_finalMask = (m_X3D.col(2).array() <= 0.f).select(0, m_finalMask);
    m_finalMask = ( (m_X3D.col(2).array()).isFinite() ).select(m_finalMask, 0);
    m_finalMask = ( (m_newX3D.row(2).transpose().array()).isFinite() ).select(m_finalMask, 0);
   
    // Check new projected x coordinates are: 0 <= x < w-1
    m_finalMask = (m_warpedX.array() < 0.f).select(0, m_finalMask);
    m_finalMask = (m_warpedX.array() >= w-2).select(0, m_finalMask);
    m_finalMask = (m_warpedX.array().isFinite()).select(m_finalMask, 0);
    // Check new projected x coordinates are: 0 <= y < h-1
    m_finalMask = (m_warpedY.array() >= h-2).select(0, m_finalMask);
    m_finalMask = (m_warpedY.array() < 0.f).select(0, m_finalMask);
    m_finalMask = (m_warpedY.array().isFinite()).select(m_finalMask, 0);

    size_t idx_cor = 0;
    size_t idx = 0;
    size_t numElements = (m_finalMask.array() != 0).count();
    float prev_distance;
    m_residual_GE.resize(2*numElements);

    euc_distance_sum = 0.f;

    std::ofstream edgeCoordinates_file;
    std::string writeFileDir = "/users/yzhen105/data/yzhen105/github_repo/GeometricError/";
    writeFileDir.append("fetch_coord_results.txt");
    edgeCoordinates_file.open(writeFileDir);
    std::cout << "numElements is " << numElements << std::endl;

    std::cout << "========================================================" << std::endl;
    std::cout << numElements << std::endl;
    std::cout << m_warpedX.rows() << std::endl;

    int reproj_indx  = 0;
    int closest_indx = 0;
    int closestX, closestY;

    for(int i = 0; i <  m_warpedX.rows(); ++i)    // new img coordinate
    {
        if(m_finalMask[i] != 0)
        {
            //prev_distance = EdgeVO::Settings::INF_F;

            //> get the index from warpedX and warpedY 
            reproj_indx  = (int)m_warpedY[i] * w + (int)m_warpedX[i];

            //> find the index of the closest from the distance transform label map
            closest_indx = m_distLabel_reference[reproj_indx]-1;

            //> recover from the closest index to pixel coordinate
            closestX = m_X2D_ref(closest_indx, 0);
            closestY = m_X2D_ref(closest_indx, 1);

            //edgeCoordinates_file << reproj_indx << "\t" << closest_indx << "\t" << (int)m_warpedX[i] << "\t" << (int)m_warpedY[i] << "\t" << closestX << "\t" << closestY << "\n";

            distance = (m_warpedX[i] - closestX)*(m_warpedX[i] - closestX) + (m_warpedY[i] - closestY)*(m_warpedY[i] - closestY);
            edgeCoordinates_file << reproj_indx << "\t" << closest_indx << "\t" << m_warpedX[i] << "\t" << m_warpedY[i] << "\t" << closestX << "\t" << closestY << "\t" << distance <<"\n";
            /*for(int j = 0; j <  m_X2D_ref.rows(); ++j)  // ref img coordinate
            {
                //std::cout << "3. run here " << std::endl;
                //* yzhen105: calculate the euclidean distance
                //distance = sqrt(pow((m_warpedX[i] - m_refX[j]),2)+pow((m_warpedY[i] - m_refX[j]),2)); //* yzhen105: #include <cmath>
                distance = (m_warpedX[i] - m_X2D_ref(j,0))*(m_warpedX[i] - m_X2D_ref(j,0))+(m_warpedY[i] - m_X2D_ref(j,1))*(m_warpedY[i] - m_X2D_ref(j,1)); //* yzhen105: #include <cmath>
                //std::cout << "4. run here " << std::endl;
                //std::cout << "distance is " << std::endl;
                //std::cout << std::endl << distance << std::endl;
                if(distance < prev_distance)
                {
                    prev_distance = distance;
                    idx_cor       = j;
                    //std::cout << "new distance is " << prev_distance << std::endl;
                    //std::cout << "with i " << i << ", j " << j << ", idx " << idx << std::endl;
                    //std::cout << "m_residual_GE[2*idx]: " << m_residual_GE[2*idx] << ", m_residual_GE[2*idx+1] " << m_residual_GE[2*idx+1] << std::endl;
                }
            }
            //edgeCoordinates_file << m_X2D_ref(idx_cor,0) << "\t" << m_X2D_ref(idx_cor,1) << "\n";
            m_residual_GE[2*idx]   = m_warpedX[i] - m_X2D_ref(idx_cor,0) ;
            m_residual_GE[2*idx+1] = m_warpedY[i] - m_X2D_ref(idx_cor,1) ;
            //std::cout << "m_warpedX.rows() is " <<  m_warpedX.rows() << ", m_X2D_ref.rows() is " << m_X2D_ref.rows() << std::endl;
            ++idx;*/
            euc_distance_sum += distance;
            //std::cout << "5. run here " << std::endl;
        }
    }
    edgeCoordinates_file.close();
    //std::cout << "6. run here " << std::endl;
    //* yzhen105: should return the sum of (x_bar-x)^2 ???error for geometric
    return (euc_distance_sum);
}

//* yzhen105: Geometric Error's Jacobian Calculation
/*void EdgeDirectVO::solveSystemOfEquationsForGE(const float lambda, const int lvl, Eigen::Matrix<double, 6 , Eigen::RowMajor>& poseupdateGE)
{
    const Mat cameraMatrix(m_sequence.getCameraMatrix(lvl));
    const float fx = cameraMatrix.at<float>(0, 0);
    const float cx = cameraMatrix.at<float>(0, 2);
    const float fy = cameraMatrix.at<float>(1, 1);
    const float cy = cameraMatrix.at<float>(1, 2);

    size_t numElements = m_im2Final.rows();
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> Z2 = m_ZFinal.array() * m_ZFinal.array();

    m_Jacobian_GE.resize(numElements, 12);
    m_Jacobian_GE.col(0)  = fx / m_ZFinal.array();
   
    m_Jacobian_GE.col(1)  = Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor>::Zero(numElements);

    m_Jacobian_GE.col(2)  = fx * m_XFinal.array() / Z2.array();

    m_Jacobian_GE.col(3)  = -fx * m_XFinal.array() * m_YFinal.array() / Z2.array();

    m_Jacobian_GE.col(4)  = fx + fx * m_XFinal.array() * m_XFinal.array() / Z2.array();

    m_Jacobian_GE.col(5)  = fx * m_YFinal.array() / m_ZFinal.array();

    m_Jacobian_GE.col(6)  = Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor>::Zero(numElements);

    m_Jacobian_GE.col(7)  = fy / m_ZFinal.array();

    m_Jacobian_GE.col(8)  = -fy * m_YFinal.array() / Z2.array();

    m_Jacobian_GE.col(9)  = -fy - fy * m_YFinal.array() * m_YFinal.array() / Z2.array();

    m_Jacobian_GE.col(10) = fy * m_XFinal.array() * m_YFinal.array() / Z2.array();

    m_Jacobian_GE.col(11) = fy * m_XFinal.array() / m_ZFinal.array();
    m_Jacobian_GE.resize(numElements*2, 6);

    //* yzhen105: ???
    //* yzhen105: N to 2*N, but how?
    //* yzhen105: possible approach
    m_weights_GE.resize(numElements*2);
    m_weights_GE = Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor>::Ones(numElements*2);
    m_weights_GE = ( ( (m_residual_GE.array()).abs() ) > EdgeVO::Settings::HUBER_THRESH ).select( 0.0 , m_weights_GE);

    m_Jacobian_GE.col(0)  = m_Jacobian_GE.col(0).array() * m_weights_GE.array();
    m_Jacobian_GE.col(1)  = m_Jacobian_GE.col(1).array() * m_weights_GE.array();
    m_Jacobian_GE.col(2)  = m_Jacobian_GE.col(2).array() * m_weights_GE.array();
    m_Jacobian_GE.col(3)  = m_Jacobian_GE.col(3).array() * m_weights_GE.array();
    m_Jacobian_GE.col(4)  = m_Jacobian_GE.col(4).array() * m_weights_GE.array();
    m_Jacobian_GE.col(5)  = m_Jacobian_GE.col(5).array() * m_weights_GE.array();
   
    m_residual_GE.array() *= m_weights_GE.array();
   
    poseupdateGE = -( (m_Jacobian_GE.transpose() * m_Jacobian_GE).cast<double>() ).ldlt().solve( (m_Jacobian_GE.transpose() * m_residual_GE).cast<double>() );

   
}*/

} //end namespace EdgeVO