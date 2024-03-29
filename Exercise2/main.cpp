#include <iostream>
#include "Eigen/Eigen"
#include <iomanip>

using namespace Eigen;
using namespace std;

//funtion for PALU decomposition:

double PALU(const MatrixXd& A, const VectorXd& b, const VectorXd& sol, Vector2d& xLU)
{
    xLU=A.partialPivLu().solve(b);
    double erroreLU = (sol-xLU).norm()/sol.norm();
    return erroreLU;
}

//funcion for QR decomposition:

double QR(const MatrixXd& A, const VectorXd& b, const VectorXd& sol, Vector2d& xQR)
{
    xQR=A.colPivHouseholderQr().solve(b);
    double erroreQR=(sol-xQR).norm()/sol.norm();
    return erroreQR;
}

int main() {

    // Define the matrices A and vectors b:
    Matrix2d A;
    Vector2d b;

    //Define optimal solution:
    Vector2d sol;
    sol << -1.0e+00,-1.0e+00;

    // System 1
    A << 5.547001962252291e-01, -3.770900990025203e-02,
        8.320502943378437e-01, -9.992887623566787e-01;
    b << -5.169911863249772e-01, 1.672384680188350e-01;

    // Solve system 1 using PALU decomposition
    Vector2d x1LU;
    double err1LU = PALU(A,b,sol,x1LU);
    cout << "System 1 using PALU decomposition:" << endl << scientific << setprecision(2) << "xLU= " << x1LU << endl << "Relative_error = " << err1LU << endl;

    // Solve system 1 using QR decomposition
    Vector2d x1QR;
    double err1QR= QR(A,b,sol,x1QR);
    cout << "System 1 using QR decomposition:" << endl << scientific << setprecision(2) << "xQR= " << x1QR << endl << " Relative_error = " << err1QR << endl;


    // System 2
    A << 5.547001962252291e-01, -5.540607316466765e-01,
        8.320502943378437e-01, -8.324762492991313e-01;
    b << -6.394645785530173e-04, 4.259549612877223e-04;

    // Solve system 2 using PALU decomposition
    Vector2d x2LU;
    double err2LU = PALU(A,b,sol,x2LU);
    cout << "System 2 using PALU decomposition:" << endl << scientific << setprecision(2) << "xLU= " << x2LU << endl << " Relative_error = " << err2LU << endl;


    // Solve system 2 using QR decomposition
    Vector2d x2QR;
    double err2QR= QR(A,b,sol,x2QR);
    cout << "System 2 using QR decomposition:" << endl << scientific << setprecision(2) << "xQR= " << x2QR << endl << " Relative_error= " << err2QR << endl;



    // System 3
    A << 5.547001962252291e-01, -5.547001955851905e-01,
        8.320502943378437e-01, -8.320502947645361e-01;
    b << -6.400391328043042e-10, 4.266924591433963e-10;

    // Solve system 3 using PALU decomposition
    Vector2d x3LU;
    double err3LU = PALU(A,b,sol,x3LU);
    cout << "System 3 using PALU decomposition:" << endl << scientific << setprecision(2) << "xLU= " << x3LU << endl << " Relative_error = " << err3LU << endl;


    // Solve system 3 using QR decomposition
    Vector2d x3QR;
    double err3QR= QR(A,b,sol,x3QR);
    cout << "System 3 using QR decomposition:" << endl << scientific << setprecision(2) << "xQR= " << x1LU << endl << " Relative_error = " << err3QR << endl;

    return 0;
}
