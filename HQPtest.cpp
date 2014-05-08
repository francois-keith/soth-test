#include "HQPtest.h"

#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

HQPSolver::HQPSolver(int size_of_q) :
	hsolver(),
	Ctasks(),
	btasks(),
	numTasks(0),
	precision(1e-3),
	nq(size_of_q)
{
}

HQPSolver::~HQPSolver()
{
	for (unsigned int i=0;i<numTasks;i++)
		delete priorities[i];
}


// Resize the problem
bool HQPSolver::configureHook(const std::vector<int> & taskSize)
{
	numTasks = taskSize.size();

	// the number of priority corresponds to the number of tasks
	priorities.resize(numTasks);

	for (unsigned int i=0;i<numTasks;i++)
		priorities[i] = new Priority(taskSize[i], nq);

	return true;
}

/* Return true iff the solver sizes fit to the task set. */
bool HQPSolver::checkSolverSize()
{
	assert( nq>0 );
	unsigned hsolverSize = numTasks;


	if(! hsolver ) return false;
	if( hsolverSize != hsolver->nbStages() ) return false;

	bool toBeResized=false;
	for( unsigned i=0;i<numTasks;++i )
	{
		assert( Ctasks[i].cols() == nq && Ctasks[i].rows() == priorities[i]->nc_priority );
		if( btasks[i].size() != priorities[i]->nc_priority)
		{
			toBeResized = true;
			break;
		}
	}

	if(( btasks[numTasks].size() != nq))
		toBeResized = true;

	return !toBeResized;
}


bool HQPSolver::solve(Eigen::VectorXd & solution)
{
	//initialize qdot
	solution.setZero();

	if(! checkSolverSize() )
		resizeSolver();

	using namespace soth;
	double damping = 1e-2;
	bool useDamping = false;
	if( useDamping ) // do we use the damping?
	{
		/* Only damp the final stage of the stack, 'cose of the solver known limitation. */
		hsolver->setDamping( 0 );
		hsolver->useDamp( true );
		hsolver->stages.back()->damping( damping );
	}
	else
	{
		hsolver->useDamp( false );
	}


	//priority loop: for each task group
	for (unsigned int i=0;i<numTasks;i++)
	{
		// Check the size of the inequalities vector.
		if(priorities[i]->inequalities.size() == 0)
			priorities[i]->inequalities.resize(0); // no inequalities.
		else if ( (priorities[i]->inequalities.size() != 0)
			&& (priorities[i]->inequalities.size() != priorities[i]->nc_priority))
				std::cerr << "Incorrect size for the vector inequalities." << std::endl;


		// -- Handle the tasks.

		// the input ports
		if(priorities[i]->Jacobian.size() == 0)
			std::cerr << "No data on A_port" << std::endl;

		if(priorities[i]->error.size() == 0)
			std::cerr << "No data on ydot_port" << std::endl;

		if(priorities[i]->error.size() == 0)
			std::cerr << "No data on ydot_port" << std::endl;

		// Fill the solver:
		// compute the jacobian.
		MatrixXd & Ctask = Ctasks[i];
		Ctask = priorities[i]->Jacobian;


		// Fill the solver: the reference.
		VectorBound & btask = btasks[i];
		const unsigned nx1 = priorities[i]->error.size();

		Eigen::VectorXd wy_ydot_lb;
		wy_ydot_lb = priorities[i]->error;

		//equality task.
		if(priorities[i]->inequalities.size() == 0)
		{
			for( unsigned c=0;c<nx1;++c )
				btask[c] = wy_ydot_lb[c];
		}
		else // inequalities task
		{
			Eigen::VectorXd wy_ydot_ub;

			// only the lower bound is given. Correct only in the case where the
			//  constraints are equality tasks or lb inequality tasks
			if(priorities[i]->error_max.size() == 0)
			{
				bool upperBoundUseless = true;
				for( unsigned c=0;c<nx1;++c )
					if ( (priorities[i]->inequalities[c] != 0) && (priorities[i]->inequalities[c] != 1))
						upperBoundUseless = false;
				if(upperBoundUseless == false)
				{
					std::cerr << "No data on ydot_max_port" << std::endl;
					std::cerr << "For now, we only handle double inequalities." << std::endl;
					return false;
				}
			}
			else
			{
				wy_ydot_ub = priorities[i]->error_max;
			}

			// Fill the solver: the error.
			for( unsigned c=0;c<nx1;++c )
			{
				switch ( priorities[i]->inequalities[c] )
				{
					case(0):
						btask[c] = wy_ydot_lb[c];
						break;
					case(1):
						btask[c] = Bound( wy_ydot_lb[c], Bound::BOUND_INF);
						break;
					case(2):
						btask[c] = Bound( wy_ydot_ub[c], Bound::BOUND_SUP);
						break;
					case(3):
						assert(wy_ydot_lb[c] <= wy_ydot_ub[c]);
						btask[c] = std::pair<double,double>(wy_ydot_lb[c], wy_ydot_ub[c]
						);
						break;
					default:
						std::cerr << " Unknown inequality type: " << priorities[i]->inequalities[c] << std::endl;
					}
			}
		}
	}

	// compute the solution
	hsolver->reset();
	hsolver->setInitialActiveSet();
	hsolver->activeSearch(solution);

	return true;
}


/** Knowing the sizes of all the stages (except the task ones),
 * the function resizes the matrix and vector of all stages
 */
void HQPSolver::resizeSolver()
{
	unsigned hsolverSize = numTasks;
	hsolver = hcod_ptr_t(new soth::HCOD( nq, hsolverSize ));
	Ctasks.resize(hsolverSize);
	btasks.resize(hsolverSize);

	for(unsigned i=0; i<numTasks;++i)
	{
		const int nx = priorities[i]->error.size();
		Ctasks[i].resize(nx,nq);
		btasks[i].resize(nx);
		hsolver->pushBackStage( Ctasks[i],btasks[i] );

		ssName.clear();
		ssName << "nc_" << i+1;
		ssName >> externalName;

		hsolver->stages.back()->name = externalName;
	}
}


HQPSolver::Priority::Priority(unsigned nc, unsigned nq)
: nc_priority(nc)
, Jacobian        (Eigen::MatrixXd::Zero(nc, nq) )
, error     (Eigen::VectorXd::Zero(nc) )
, error_max (Eigen::VectorXd::Zero(nc) )
, inequalities (0)
{
}


