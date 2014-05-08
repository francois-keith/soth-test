#ifndef _HQP_SOLVER_HPP
#define HQP_SOLVER_HPP

#include <algorithm>
#include <soth/HCOD.hpp>
#include <Eigen/LU>

#include <vector>
#include <string>
#include <sstream>

class HQPSolver
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	HQPSolver(int size_of_q);
	virtual ~HQPSolver();//destructor

	virtual bool configureHook(const std::vector<int> & taskSize);
	virtual bool solve(Eigen::VectorXd & solution);

	void resizeSolver();
	bool checkSolverSize();

	//unsigned int nc;
	unsigned int nq;
	bool inequalityProvisions;

private:
	// the solver
	typedef boost::shared_ptr<soth::HCOD> hcod_ptr_t;
	hcod_ptr_t hsolver;

	// Jacobian of the tasks
	std::vector< Eigen::MatrixXd > Ctasks;

	// Upper/ Lower bounds for each tasks
	std::vector< soth::VectorBound > btasks;

	// attribute: number of tasks involved (default = 1)
	unsigned int numTasks;

	// property: precision of matrix comparison
	double precision;


public:
	//std::vector priorities contains structs of the "Priority" type
  // A Priority corresponds to a layer of the HQP (aka a minimization problem).
	struct Priority
	{
	public:
		// nq is the number of joints
		// nc is the size of the error
		Priority(unsigned nc, unsigned nq);

	public:
		//// PARAMETERS
		// error size
		unsigned int nc_priority;

		//generalized jacobian for a subtask with a certain priority
		Eigen::MatrixXd Jacobian;

		//task space coordinates
		Eigen::VectorXd error;

		//Upper bound desired task value (optional)
		Eigen::VectorXd error_max;

		// vector indicating the type of constraint considered (optional)
		// 0: equality,  1: lower inequality, 2: upper inequality,
		// 3: double bound inequality
		std::vector<unsigned> inequalities;
	};

	std::vector<Priority*> priorities;

// temporary data only used to avoid time consuption at the creation.
private:
	std::string externalName;
	std::stringstream ssName;
};//end class definition

#endif
