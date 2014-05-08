#include "HQPtest.h"

// Specify one test
int main()
{
  // Size of the parameter vector
	int nq = 7;
	HQPSolver solver(nq);

	// we solve a system with 2 tasks
  // first of size 1
  // second of size 6 (6 constraints)
	std::vector <int> taskSize(2);
	taskSize[0] = 1;
	taskSize[1] = 6;
	solver.configureHook(taskSize);

	// define the first task (inequality)
	int error1Dimension = taskSize[0];
	solver.priorities[0]->Jacobian.resize(error1Dimension, nq);
	solver.priorities[0]->Jacobian << 1,	0,	0,	0,	0,	0,	0;

	solver.priorities[0]->error.resize(error1Dimension);
	solver.priorities[0]->error << 2;

	solver.priorities[0]->error_max.resize(error1Dimension);
	solver.priorities[0]->error_max << 100;

	//the following resize activates the inequality mode.
	solver.priorities[0]->inequalities.resize(error1Dimension);
	solver.priorities[0]->inequalities[0] = 1;



	// define the second task (equality)
	int error2Dimension = taskSize[1];
	solver.priorities[1]->Jacobian.resize(error2Dimension, nq);
	solver.priorities[1]->Jacobian<<
	1  , 	1.3,	3.9,	6.9,	0,	1.9,	9.4,
	6.9,	2.2,	1.7,	0,	3.1,	2.4,	4.7,
	0  ,	8.8,	1.9,	0,	0,	0,	1.9,
	0  ,	3.2,	1.9,	0,	0,	0,	2.3,
	2.2,	1.5,	0,	5.5,	2.1,	0,	2.1,
	5.2,	7.4,	0,	0,	3.1,	9.2,	1.0;

	solver.priorities[1]->error.resize(error2Dimension);
	solver.priorities[1]->error << 2,	1.3,	2.1,	0.2,	2.3, 	0.9;

	solver.priorities[1]->error_max.resize(0);

	Eigen::VectorXd solution(nq);
	solver.solve(solution);

	Eigen::VectorXd qdot_expected(nq);
	qdot_expected <<  2,
			0.237781995468, 1.42500913707, 1.12909026746,
			-2.7059333957, -0.157624040907, -1.42105206345;

	std::cerr << solution.transpose() << std::endl;
	std::cerr << qdot_expected.transpose() << std::endl;
	return 1;
}
