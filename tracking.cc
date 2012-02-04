#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

static double const HOUGH_RHO    = 1;
static double const HOUGH_THETA  = CV_PI / 180;
static int    const HOUGH_THRESH = 10;

int main(int argc, char *argv[])
{
    if (argc <= 1) {
        std::cerr << "err: incorrect number of arguments\n"
                  << "usage: ./tracking <image>"
                  << std::endl;
        return 1;
    }

    std::string path = argv[1];
    cv::Mat color = cv::imread(path, 1);

    cv::Mat gray;
    cv::cvtColor(color, gray, CV_BGR2GRAY);

    cv::Mat edges;
    cv::Canny(gray, edges, 50, 200, 3);

    int const rows = color.rows;
    int const cols = color.cols;

    // Approximate contours as four-sided polygons.
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(edges, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point> > polygons(contours.size());

    for (size_t i = 0; i < contours.size(); i++) {
        std::vector<cv::Point> contour = contours[i];
        double length = cv::arcLength(contour, true);

        std::vector<cv::Point> polygon;
        cv::approxPolyDP(contours[i], polygon, length * 0.02, true);
        polygons[i] = polygon;
    }

    // Ignore the inner-most polygons and those without four vertices.
    for (size_t i = 0; i < polygons.size(); i++) {
        std::vector<cv::Point> outer = polygons[i];
        if (outer.size() != 4) continue;

        // Search for first inner polygon with four corners.
        for (int j = hierarchy[i][2]; j > 0; j = hierarchy[j][1]) {
            std::cout << "outer = " << i << ", inner = " << j << std::endl;
            std::vector<cv::Point> inner = polygons[j];
            if (inner.size() != 4) continue;

            // Mask the area between the inner and outer polygons.
            cv::Mat mask_inner(rows, cols, CV_8U, cv::Scalar(0));
            cv::Mat mask_outer(rows, cols, CV_8U, cv::Scalar(0));
            cv::fillConvexPoly(mask_inner, &inner[0], inner.size(), 255);
            cv::fillConvexPoly(mask_outer, &outer[0], outer.size(), 255);
            cv::Mat mask = mask_outer - mask_inner;

// DEBUG
            cv::Mat render;
            color.copyTo(render);
            render.setTo(cv::Scalar(0, 0, 255), mask);

            cv::imshow("mask", render);
            while (cv::waitKey() != ' ');


#if 0
            std::vector<std::vector<cv::Point> > contour_render(2);
            contour_render[0] = inner;
            contour_render[1] = outer;

            cv::drawContours(render, contour_render, 0, cv::Scalar(0, 0, 255)); 
            cv::drawContours(render, contour_render, 1, cv::Scalar(0, 255, 0)); 

            cv::imshow("contour pair", render);
            while (cv::waitKey() != ' ');
#endif
// DEBUG
        }
    }
    return 0;
}
