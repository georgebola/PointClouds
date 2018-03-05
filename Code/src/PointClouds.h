#include <VVRScene/canvas.h>
#include <VVRScene/mesh.h>
#include <MathGeoLib.h>
#include <opencv2/highgui.hpp>

#include <eigen-eigen-67e894c6cd8f/Eigen/Dense>
#include <eigen-eigen-67e894c6cd8f/Eigen/SVD>

namespace vvr {
	typedef std::vector<vvr::Point3D> PointCloud;
	std::string imgDir = "../resources/images/";
}

namespace Eigen {
	
}


/**
* To save typing.
*/
typedef std::vector<vec> VecVector;


/**
* A node of a KD-Tree
*/
struct KDNode
{
	vec split_point;
	int axis;
	int level;
	AABB aabb;
	KDNode *child_left;
	KDNode *child_right;
	KDNode() : child_left(NULL), child_right(NULL) {}
	~KDNode() { delete child_left; delete child_right; }
};

/**
* KD-Tree wrapper. Holds a ptr to tree root.
*/
class KDree
{
public:
	KDree(VecVector &pts);
	~KDree();
	std::vector<KDNode*> getNodesOfLevel(int level);
	int depth() const { return m_depth; }
	const KDNode* root() const { return m_root; }
	const VecVector &pts;

private:
	static int makeNode(KDNode *node, VecVector &pts, const int level);
	static void getNodesOfLevel(KDNode *node, std::vector<KDNode*> &nodes, int level);

private:
	KDNode *m_root;
	int m_depth;
};

class PointCloudScene : public vvr::Scene
{
	// Simple bit-controlled enumerator
	enum
	{
		FLAG_SHOW_SOLID = 1   << 0,
		FLAG_SHOW_NORMALS = 1 << 1,
		FLAG_SHOW_WIRE = 1    << 2,
		FLAG_SHOW_AXES = 1    << 3,
		FLAG_SHOW_POINTS = 1  << 4,
		FLAG_SHOW_DISTANCES = 1   << 5,
	};
public:
    PointCloudScene();
    const char* getName() const { return "PointCloud Scene"; }
	void keyEvent(unsigned char key, bool up, int modif) override;
    void arrowEvent(vvr::ArrowDir dir, int modif) override;

private:
    void draw() override;
    void reset() override;
    void resize() override;

	float m_tree_invalidation_sec;

	// Helper method to get and draw PointCloud from Mesh
	void getPointCloud(std::vector<cv::Mat> &depths, std::vector<std::vector<vec>> &m_pointclouds);
	void getSobelPointCloud(std::vector<cv::Mat> &im_sobel, std::vector<std::vector<vec>> &sobel_pointclouds, float thresholding);

	void drawPointCloud(std::vector<vec> m_vertices, const vvr::Colour& color);
	void drawColouredPointCloud(std::vector<vec> m_vertices, std::vector<vec> &m_colours);

	void closest_neighbor(int NUMBER_OF_IMAGE, std::vector<std::vector<vec>> &m_pointclouds, std::vector<vec> &close_points);
	void rotation_translation(int NUMBER_OF_IMAGE, std::vector<std::vector<vec>> &m_pointclouds, Eigen::MatrixXf &R, Eigen::MatrixXf &t);
	
	//void KDtree_closest(int image, vector<vector<vec>> &m_pointclouds, vector<vec> &close_points);
	
	void sobel(cv::Mat &image, cv::Mat &new_image);
	
	void many_p(const cv::Mat& image, std::vector<vec> &vertices, std::vector<std::vector<vec>> &m_many_pclouds, std::vector<float> &distances);

	void TriangulateMesh(const std::vector<vec> &vertices, vvr::Mesh*& mesh, float dist);

	//, Eigen::MatrixXf &R, Eigen::MatrixXf &t);
	// Members
    int m_style_flag;
	// desk_1 , desk_2 , table_1 , table_small_1
	// 20 , 40 , 10 , 25
	int count = 90;
	int step = 10;
	int image_from = 19;
	int image_to = 20;
	int num_pointcloud = 20;
	int NUMBER_OF_IMAGE = image_to;

	KDree *m_KDree;
	vec nn;
	vec nn1;

	vec nn2;


    float m_plane_d;
    vvr::Canvas2D m_canvas;
    vvr::Colour m_point_color;
    vvr::Mesh m_model_original;
    math::Plane m_plane;
	vvr::PointCloud m_model_points;
	std::vector<vvr::Mesh*> m_model;

	std::vector<std::vector<vec>> m_colours;
	std::vector<vec> m_rotated;

	std::vector<std::vector<vec>> m_pointcloud_triang;
	std::vector<std::vector<vec>> m_many_pclouds;
	std::vector<std::vector<vec>> m_rot_pointclouds;
	std::vector<std::vector<vec>> m_pointclouds;
	std::vector<std::vector<vec>> m_sobel_pointclouds;

	std::vector<std::vector<vec>> sobel_pointclouds;
	std::vector<std::vector<vec>> m_new;
	std::vector<vec> m_close_points;
	std::vector<vec> m_POINTCLOUD;
	std::vector<vec> no_duplicates;
	std::vector<cv::Mat> im_sobel;
	std::vector<Eigen::MatrixXf> all_rots;
	std::vector<Eigen::MatrixXf> all_trans;

	// example depth image
	cv::Mat m_depth_image, m_color_image, m_depth_image_norm,sobelx;

	std::vector<cv::Mat> images;
	std::vector<cv::Mat> depths;
};

struct VecComparator {
	unsigned axis;
	VecComparator(unsigned axis) : axis(axis % 3) {}
	virtual inline bool operator() (const vec& v1, const vec& v2) {
		return (v1.ptr()[axis] < v2.ptr()[axis]);
	}
};

void KD_Nearest(const vec& test_pt, const KDNode* root, const KDNode* &nn, float& best_dist);
