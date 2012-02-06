#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

struct Target {
    std::vector<cv::Point> polygon_inner;
    std::vector<cv::Point> polygon_outer;
    cv::Mat mask;
    double score;
    bool dominant;
};

static double const THRESHOLD = 100;
static double const THRESHOLD_AREA = 500.0;

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

    std::vector<std::vector<cv::Point> > polygons;

    for (size_t i = 0; i < contours.size(); i++) {
        std::vector<cv::Point> contour = contours[i];
        double length = cv::arcLength(contour, true);

        std::vector<cv::Point> polygon;
        cv::approxPolyDP(contours[i], polygon, length * 0.02, true);
        polygons.push_back(polygon);
    }

    // Ignore the inner-most polygons and those without four vertices.
    std::vector<Target> targets;
    std::vector<double> scores;

    for (size_t i = 0; i < polygons.size(); i++) {
        std::vector<cv::Point> outer = polygons[i];
        if (outer.size() != 4) continue;

        // Search for first inner polygon with four corners.
        for (int j = hierarchy[i][2]; j > 0; j = hierarchy[j][1]) {
            std::vector<cv::Point> inner = polygons[j];
            if (inner.size() != 4) continue;

            // Mask the area between the inner and outer polygons.
            cv::Mat mask_inner(rows, cols, CV_8U, cv::Scalar(0));
            cv::Mat mask_outer(rows, cols, CV_8U, cv::Scalar(0));
            cv::fillConvexPoly(mask_inner, &inner[0], inner.size(), 255);
            cv::fillConvexPoly(mask_outer, &outer[0], outer.size(), 255);
            cv::Mat mask = mask_outer - mask_inner;

            // Compute the average brightness in the masked region.
            // TODO: Add a parallel line constraint.
            // TODO: Replace this with histogram matching to improve accuracy.
            cv::Scalar mean = cv::mean(color, mask);

            Target target;
            target.polygon_inner = inner;
            target.polygon_outer = outer;
            target.mask = mask;
            target.score = cv::norm(mean);
            target.dominant = true;
            targets.push_back(target);
        }
    }

    // Check which targets intersect.
    // TODO: Make this more efficient by using a disjoint-sets data structure.
    std::vector<bool> dominant_flags(targets.size(), true);

    for (size_t i = 0; i < targets.size(); i++) {
        Target &ti = targets[i];

        for (size_t j = i + 1; j < targets.size(); j++) {
            Target &tj = targets[j];

            double max_overlap;
            cv::Mat overlap = cv::min(ti.mask, tj.mask);
            cv::minMaxLoc(overlap, NULL, &max_overlap);

            if (max_overlap) {
                if (ti.score > tj.score) {
                    tj.dominant = false;
                } else {
                    ti.dominant = false;
                }
            }
        }
    }

    // Filter out non-dominant targets.
    std::vector<Target> dominant_targets;

    for (size_t i = 0; i < targets.size(); i++) {
        if (targets[i].dominant) {
            dominant_targets.push_back(targets[i]);
        }
    }

    // Overlay the targets.
    for (size_t i = 0; i < dominant_targets.size(); i++) {
        cv::Mat render;
        color.copyTo(render);
        render.setTo(cv::Scalar(0, 255, 0), dominant_targets[i].mask);

        double inner_area = cv::contourArea(dominant_targets[i].polygon_inner, false);
        double outer_area = cv::contourArea(dominant_targets[i].polygon_outer, false);
        double area = outer_area - inner_area;

        if (area >= THRESHOLD_AREA) {
            cv::imshow("targets", render);
            while (cv::waitKey() != ' ');
        }
    }
    return 0;
}
