// gubas_core.cpp — F2BP simulation library
// Physics from Hou 2016 / GUBAS.  Units: km, kg, s.
#define _USE_MATH_DEFINES
#include "gubas_core.h"
#include <iostream>
#include <cmath>
#include <stdexcept>
using namespace std;
using namespace arma;

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

double factorial(double x){
    double fact=1.;
    while(x>1){ fact*=(double)x; x--; }
    return fact;
}

mat tilde_op(double x, double y, double z){
    mat tilde(3,3);
    tilde<<0.<<-z<<y<<endr<<z<<0.<<-x<<endr<<-y<<x<<0.<<endr;
    return tilde;
}

mat tilde_opv(mat A){
    mat tilde(3,3);
    tilde<<0.<<-A(2,0)<<A(1,0)<<endr<<A(2,0)<<0.<<-A(0,0)<<endr<<-A(1,0)<<A(0,0)<<0.<<endr;
    return tilde;
}

// ─────────────────────────────────────────────────────────────────────────────
// Expansion coefficients
// ─────────────────────────────────────────────────────────────────────────────

void tk_calc(int m, mat* t){
    // Compute the t_{n,k} expansion coefficients from Hou 2016 Table 1.
    // t is stored as a matrix: row n, column k/2 (k steps by 2 from fmod(n,2)).
    for(int n=0;n<m+1;n++){
        // Seed value t_{n,0}: closed-form differs for odd vs. even n
        if(n%2){
            (*t)(n,0)=pow(-1.,(n-1.)/2.)*factorial((double)n)/(pow(2.,n-1.)*pow(factorial((n-1.)/2.),2.));
        } else {
            (*t)(n,0)=pow(-1.,n/2.)*factorial((double)n)/(pow(2.,n)*pow(factorial(n/2.),2.));
        }
        // Recursion: t_{n,k+2} = -(n-k)(n+k+1)/((k+2)(k+1)) * t_{n,k}
        // k starts at fmod(n,2) so only even or odd k values are filled for a given n
        double k=fmod(n,2.);
        int i=1;
        while(k<=n){
            (*t)(n,i)=-(n-k)*(n+k+1.)*((*t)(n,i-1))/((k+2.)*(k+1.));
            k+=2.; i++;
        }
    }
}

int t_ind(int a, int b, int c, int d, int e, int f, int g, int dim){
    int ind;
    if(a+b+c+d+e+f+g>pow(dim,7)){
        ind=(dim-a)*pow(dim,6)+(dim-b)*pow(dim,5)+(dim-c)*pow(dim,4)+(dim-d)*pow(dim,3)+(dim-e)*pow(dim,2)+(dim-f)*dim+g;
    } else {
        ind=a*pow(dim,6)+b*pow(dim,5)+c*pow(dim,4)+d*pow(dim,3)+e*pow(dim,2)+f*dim+g;
    }
    return ind;
}

void a_calc(int n, sp_mat* a){
    // Compute a_{k,i1,i2,i3,i4,i5,i6} expansion coefficients from Hou 2016.
    // These encode the multinomial expansion of (e·r_A)^k * (e·r_B)^(n-k)
    // where i1,i2,i3 count x,y,z contributions from body A and i4,i5,i6 from body B.
    // The 7-D index set is flattened to 1-D via t_ind().
    (*a)(0,0)=1;
    if(n>0){
        // Seed: order-1 coefficients by inspection from Hou
        (*a)(0,t_ind(1,1,0,0,0,0,0,n+1))=1;
        (*a)(0,t_ind(1,0,1,0,0,0,0,n+1))=1;
        (*a)(0,t_ind(1,0,0,1,0,0,0,n+1))=1;
        (*a)(0,t_ind(1,0,0,0,1,0,0,n+1))=-1;
        (*a)(0,t_ind(1,0,0,0,0,1,0,n+1))=-1;
        (*a)(0,t_ind(1,0,0,0,0,0,1,n+1))=-1;
        if(n>1){
            // Build higher-order coefficients via the Hou recursion.
            // Outer loop: expansion order k from 2 up to n.
            for(int k=2;k<n+1;k++){
                // Inner 6 loops enumerate all index tuples (i1..i6) with i1+..+i6 <= k.
                // i6 is the "remainder" index: i6 = k - i1 - i2 - i3 - i4 - i5,
                // so the loop upper bounds enforce i1+..+i6 == k exactly.
                for(int i1=0;i1<k+1;i1++){
                    for(int i2=0;i2<k+1-i1;i2++){
                        for(int i3=0;i3<k+1-i1-i2;i3++){
                            for(int i4=0;i4<k+1-i1-i2-i3;i4++){
                                for(int i5=0;i5<k+1-i1-i2-i3-i4;i5++){
                                    for(int i6=0;i6<k+1-i1-i2-i3-i4-i5;i6++){
                                        // Recursion: a(k,i) = sum over each index that can be
                                        // decremented by 1, adding or subtracting the order-(k-1)
                                        // coefficient. i1,i2,i3 (body A) contribute positively;
                                        // i4,i5,i6 (body B / cross terms) contribute negatively.
                                        if(i1>0) (*a)(0,t_ind(k,i1,i2,i3,i4,i5,i6,n+1))+=(*a)(0,t_ind(k-1,i1-1,i2,i3,i4,i5,i6,n+1));
                                        if(i2>0) (*a)(0,t_ind(k,i1,i2,i3,i4,i5,i6,n+1))+=(*a)(0,t_ind(k-1,i1,i2-1,i3,i4,i5,i6,n+1));
                                        if(i3>0) (*a)(0,t_ind(k,i1,i2,i3,i4,i5,i6,n+1))+=(*a)(0,t_ind(k-1,i1,i2,i3-1,i4,i5,i6,n+1));
                                        if(i4>0) (*a)(0,t_ind(k,i1,i2,i3,i4,i5,i6,n+1))-=(*a)(0,t_ind(k-1,i1,i2,i3,i4-1,i5,i6,n+1));
                                        if(i5>0) (*a)(0,t_ind(k,i1,i2,i3,i4,i5,i6,n+1))-=(*a)(0,t_ind(k-1,i1,i2,i3,i4,i5-1,i6,n+1));
                                        if(i6>0) (*a)(0,t_ind(k,i1,i2,i3,i4,i5,i6,n+1))-=(*a)(0,t_ind(k-1,i1,i2,i3,i4,i5,i6-1,n+1));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void b_calc(int n, sp_mat* b){
    // Compute b_{k,j1..j6} expansion coefficients from Hou 2016.
    // The b coefficients encode the Laplacian-based correction terms that appear
    // in the mutual potential when expanding 1/|r|^n.  They satisfy a different
    // recursion from the a coefficients: b(m) depends on b(m-2) (steps of 2),
    // which is why k descends from n and the (n-k)>2 guard skips the seeded orders.
    (*b)(0,0)=1;
    if(n>1){
        // Seed: order-2 coefficients set directly from Hou
        (*b)(0,t_ind(2,2,0,0,0,0,0,n+1))=1;
        (*b)(0,t_ind(2,0,2,0,0,0,0,n+1))=1;
        (*b)(0,t_ind(2,0,0,2,0,0,0,n+1))=1;
        (*b)(0,t_ind(2,0,0,0,2,0,0,n+1))=1;
        (*b)(0,t_ind(2,0,0,0,0,2,0,n+1))=1;
        (*b)(0,t_ind(2,0,0,0,0,0,2,n+1))=1;
        (*b)(0,t_ind(2,1,0,0,1,0,0,n+1))=-2;
        (*b)(0,t_ind(2,0,1,0,0,1,0,n+1))=-2;
        (*b)(0,t_ind(2,0,0,1,0,0,1,n+1))=-2;
        // Outer loop descends k from n to 0; the effective expansion order is (n-k).
        // Orders 0 and 2 are already seeded above, so the recursion only fires for (n-k)>2.
        for(int k=n;k>-1;k--){
            // Inner 6 loops enumerate index tuples (j1..j6) with j1+..+j6 <= (n-k),
            // same flattening convention as a_calc.
            for(int j1=0;j1<n-k+1;j1++){
                for(int j2=0;j2<n-k+1-j1;j2++){
                    for(int j3=0;j3<n-k+1-j1-j2;j3++){
                        for(int j4=0;j4<n-k+1-j1-j2-j3;j4++){
                            for(int j5=0;j5<n-k+1-j1-j2-j3-j4;j5++){
                                for(int j6=0;j6<n-k+1-j1-j2-j3-j4-j5;j6++){
                                    if((n-k)>2){
                                        // Recursion: cross terms (j_i>0 && j_{i+3}>0) subtract 2× b at order-2,
                                        // diagonal terms (j_i>1) add b at order-2.
                                        // This mirrors the action of the Laplacian operator on the polynomial basis.
                                        if(j1>0&&j4>0) (*b)(0,t_ind(n-k,j1,j2,j3,j4,j5,j6,n+1))+=-2*(*b)(0,t_ind(n-k-2,j1-1,j2,j3,j4-1,j5,j6,n+1));
                                        if(j2>0&&j5>0) (*b)(0,t_ind(n-k,j1,j2,j3,j4,j5,j6,n+1))+=-2*(*b)(0,t_ind(n-k-2,j1,j2-1,j3,j4,j5-1,j6,n+1));
                                        if(j3>0&&j6>0) (*b)(0,t_ind(n-k,j1,j2,j3,j4,j5,j6,n+1))+=-2*(*b)(0,t_ind(n-k-2,j1,j2,j3-1,j4,j5,j6-1,n+1));
                                        if(j1>1) (*b)(0,t_ind(n-k,j1,j2,j3,j4,j5,j6,n+1))+=(*b)(0,t_ind(n-k-2,j1-2,j2,j3,j4,j5,j6,n+1));
                                        if(j2>1) (*b)(0,t_ind(n-k,j1,j2,j3,j4,j5,j6,n+1))+=(*b)(0,t_ind(n-k-2,j1,j2-2,j3,j4,j5,j6,n+1));
                                        if(j3>1) (*b)(0,t_ind(n-k,j1,j2,j3,j4,j5,j6,n+1))+=(*b)(0,t_ind(n-k-2,j1,j2,j3-2,j4,j5,j6,n+1));
                                        if(j4>1) (*b)(0,t_ind(n-k,j1,j2,j3,j4,j5,j6,n+1))+=(*b)(0,t_ind(n-k-2,j1,j2,j3,j4-2,j5,j6,n+1));
                                        if(j5>1) (*b)(0,t_ind(n-k,j1,j2,j3,j4,j5,j6,n+1))+=(*b)(0,t_ind(n-k-2,j1,j2,j3,j4,j5-2,j6,n+1));
                                        if(j6>1) (*b)(0,t_ind(n-k,j1,j2,j3,j4,j5,j6,n+1))+=(*b)(0,t_ind(n-k-2,j1,j2,j3,j4,j5,j6-2,n+1));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Inertia integrals
// ─────────────────────────────────────────────────────────────────────────────

double Q_ijk(double i, double j, double k){
    return factorial(i)*factorial(j)*factorial(k)/factorial(i+j+k+3.);
}

double tet_sums(double l, double m, double n,
                double x1, double x2, double x3,
                double y1, double y2, double y3,
                double z1, double z2, double z3){
    // Compute the tetrahedron summation for inertia integral T_{l,m,n} from Hou 2016.
    // A tetrahedron has one vertex at the origin and three others at (x1,y1,z1),
    // (x2,y2,z2), (x3,y3,z3).  The integral of x^l * y^m * z^n over this tet has
    // a closed form as a triple multinomial sum (see Hou eq. for T_lmn).
    //
    // Each pair of loops (i1,j1), (i2,j2), (i3,j3) is a multinomial expansion
    // of (x1+x2+x3)^l, (y1+y2+y3)^m, (z1+z2+z3)^n respectively.
    // The "remainder" exponent in each triplet is (l-i1-j1), (m-i2-j2), (n-i3-j3).
    // Q_ijk then provides the weighting factor from the integral over the standard simplex.
    double sum_val=0.;
    for(double i1=0.;i1<(l+1.);i1++){
        for(double j1=0.;j1<(l-i1+1.);j1++){
            for(double i2=0.;i2<(m+1.);i2++){
                for(double j2=0.;j2<(m-i2+1.);j2++){
                    for(double i3=0.;i3<(n+1.);i3++){
                        for(double j3=0.;j3<(n-i3+1.);j3++){
                            sum_val+=(factorial(l)/(factorial(i1)*factorial(j1)*factorial(l-i1-j1)))
                                    *(factorial(m)/(factorial(i2)*factorial(j2)*factorial(m-i2-j2)))
                                    *(factorial(n)/(factorial(i3)*factorial(j3)*factorial(n-i3-j3)))
                                    *pow(x1,i1)*pow(x2,j1)*pow(x3,(l-i1-j1))
                                    *pow(y1,i2)*pow(y2,j2)*pow(y3,(m-i2-j2))
                                    *pow(z1,i3)*pow(z2,j3)*pow(z3,(n-i3-j3))
                                    *Q_ijk(i1+i2+i3,j1+j2+j3,l+m+n-i1-i2-i3-j1-j2-j3);
                        }
                    }
                }
            }
        }
    }
    return sum_val;
}

void poly_inertia_met(int q, double rho, string tet_file, string vert_file, cube* T){
    mat tet,vert,x1,x2,x3,x;
    double Ta;
    (*T).zeros(q+1,q+1,q+1);
    tet.load(tet_file,csv_ascii);
    tet-=1;
    vert.load(vert_file,csv_ascii);
    vert=vert/1000.;
    for(double l=0;l<(q+1);l++){
        for(double m=0;m<(q+1-l);m++){
            for(double n=0;n<(q+1-l-m);n++){
                for(int a=0;a<(int)tet.n_rows;a++){
                    x1=vert((int)tet(a,0),span(1,3));
                    x2=vert((int)tet(a,1),span(1,3));
                    x3=vert((int)tet(a,2),span(1,3));
                    x=join_vert(x1,x2);
                    x=join_vert(x,x3);
                    Ta=abs(det(x));
                    (*T)(l,m,n)+=rho*Ta*tet_sums(l,m,n,x1(0,0),x2(0,0),x3(0,0),
                                                  x1(0,1),x2(0,1),x3(0,1),
                                                  x1(0,2),x2(0,2),x3(0,2));
                }
            }
        }
    }
}

void inertia_rot(mat C, int q, cube* T, cube* Tp){
    // Rotate the inertia integral set T (principal-frame of body B) into the A frame
    // using rotation matrix C (transforms B coords to A coords).
    // The rotation formula expands each monomial x^l*y^m*z^n in terms of the rotated
    // coordinates using the multinomial theorem applied to each row of C.
    // See Hou 2016 eq. for T'_{l,m,n}.
    (*Tp).zeros(q+1,q+1,q+1);
    // Outer triple loop: iterate over every (l,m,n) output inertia integral index
    for(int l=0;l<(q+1);l++){
        for(int m=0;m<(q+1-l);m++){
            for(int n=0;n<(q+1-l-m);n++){
                // Inner 6 loops: multinomial expansion for each row of C.
                // Row 0 of C: C(0,0)^i1 * C(0,1)^j1 * C(0,2)^(l-i1-j1)
                // Row 1 of C: C(1,0)^i2 * C(1,1)^j2 * C(1,2)^(m-i2-j2)
                // Row 2 of C: C(2,0)^i3 * C(2,1)^j3 * C(2,2)^(n-i3-j3)
                // These index the original (unrotated) inertia integral T at
                // (i1+i2+i3, j1+j2+j3, remainder), hence the constraint check below.
                for(double i1=0.;i1<(l+1.);i1++){
                    for(double j1=0.;j1<(l-i1+1.);j1++){
                        for(double i2=0.;i2<(m+1.);i2++){
                            for(double j2=0.;j2<(m-i2+1.);j2++){
                                for(double i3=0.;i3<(n+1.);i3++){
                                    for(double j3=0.;j3<(n-i3+1.);j3++){
                                        // Guard: combined indices must stay within the truncated cube
                                        if(((i1+i2+i3)<=q)&&((j1+j2+j3)<=q)&&((l+m+n-i1-i2-i3-j1-j2-j3)<=q)){
                                            (*Tp)(l,m,n)+=(factorial((double)l)/(factorial((double)i1)*factorial((double)j1)*factorial((double)l-i1-j1)))
                                                *(factorial((double)m)/(factorial((double)i2)*factorial((double)j2)*factorial((double)(m-i2-j2))))
                                                *(factorial((double)n)/(factorial((double)i3)*factorial((double)j3)*factorial((double)(n-i3-j3))))
                                                *pow(C(0,0),i1)*pow(C(0,1),j1)*pow(C(0,2),(l-i1-j1))
                                                *pow(C(1,0),i2)*pow(C(1,1),j2)*pow(C(1,2),(m-i2-j2))
                                                *pow(C(2,0),i3)*pow(C(2,1),j3)*pow(C(2,2),(n-i3-j3))
                                                *(*T)(i1+i2+i3,j1+j2+j3,l+m+n-i1-i2-i3-j1-j2-j3);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void poly_moi_met(double rho, string tet_file, string vert_file, mat* I){
    mat tet,vert;
    tet.load(tet_file,csv_ascii);
    tet-=1;
    vert.load(vert_file,csv_ascii);
    vert=vert/1000.;
    double V;
    mat p1,p2,p3,p4,p;
    for(int a=0;a<(int)tet.n_rows;a++){
        p1.zeros(1,3);
        p2=vert((int)tet(a,0),span(1,3));
        p3=vert((int)tet(a,1),span(1,3));
        p4=vert((int)tet(a,2),span(1,3));
        p=join_vert(p2,p3);
        p=join_vert(p,p4);
        V=rho*abs(det(p))/6.;
        (*I)(0,0)+=V*(pow(p1(0,1),2)+p1(0,1)*p2(0,1)+pow(p2(0,1),2)+p1(0,1)*p3(0,1)+p2(0,1)*p3(0,1)+pow(p3(0,1),2)+
                pow(p1(0,2),2)+p1(0,2)*p2(0,2)+pow(p2(0,2),2)+p1(0,2)*p3(0,2)+p2(0,2)*p3(0,2)+pow(p3(0,2),2)+
                p1(0,1)*p4(0,1)+p2(0,1)*p4(0,1)+p3(0,1)*p4(0,1)+pow(p4(0,1),2)+
                p1(0,2)*p4(0,2)+p2(0,2)*p4(0,2)+p3(0,2)*p4(0,2)+pow(p4(0,2),2))/10.;
        (*I)(0,1)+=V*(pow(p1(0,0),2)+p1(0,0)*p2(0,0)+pow(p2(0,0),2)+p1(0,0)*p3(0,0)+p2(0,0)*p3(0,0)+pow(p3(0,0),2)+
                pow(p1(0,2),2)+p1(0,2)*p2(0,2)+pow(p2(0,2),2)+p1(0,2)*p3(0,2)+p2(0,2)*p3(0,2)+pow(p3(0,2),2)+
                p1(0,0)*p4(0,0)+p2(0,0)*p4(0,0)+p3(0,0)*p4(0,0)+pow(p4(0,0),2)+
                p1(0,2)*p4(0,2)+p2(0,2)*p4(0,2)+p3(0,2)*p4(0,2)+pow(p4(0,2),2))/10.;
        (*I)(0,2)+=V*(pow(p1(0,1),2)+p1(0,1)*p2(0,1)+pow(p2(0,1),2)+p1(0,1)*p3(0,1)+p2(0,1)*p3(0,1)+pow(p3(0,1),2)+
                pow(p1(0,0),2)+p1(0,0)*p2(0,0)+pow(p2(0,0),2)+p1(0,0)*p3(0,0)+p2(0,0)*p3(0,0)+pow(p3(0,0),2)+
                p1(0,1)*p4(0,1)+p2(0,1)*p4(0,1)+p3(0,1)*p4(0,1)+pow(p4(0,1),2)+
                p1(0,0)*p4(0,0)+p2(0,0)*p4(0,0)+p3(0,0)*p4(0,0)+pow(p4(0,0),2))/10.;
    }
}

void ell_mass_params_met(double order, double order_body, double rho,
                         double a, double b, double c, mat* I, cube* T){
    // a, b, c are already in km (Python converts m→km before calling run_simulation)
    double M=4.*rho*M_PI*a*b*c/3.;
    (*T).zeros(order+1,order+1,order+1);
    (*T)(0,0,0)=M;
    (*I)(0,0)=M*(pow(b,2)+pow(c,2))/5.;
    (*I)(0,1)=M*(pow(a,2)+pow(c,2))/5.;
    (*I)(0,2)=M*(pow(b,2)+pow(a,2))/5.;
    if(order_body>0){
        (*T)(2,0,0)=M*pow(a,2)/5.;
        (*T)(0,2,0)=M*pow(b,2)/5.;
        (*T)(0,0,2)=M*pow(c,2)/5.;
        if(order_body>3){
            (*T)(4,0,0)=3.*M*pow(a,4)/35.;
            (*T)(0,4,0)=3.*M*pow(b,4)/35.;
            (*T)(0,0,4)=3.*M*pow(c,4)/35.;
            (*T)(2,2,0)=M*pow(a,2)*pow(b,2)/35.;
            (*T)(0,2,2)=M*pow(c,2)*pow(b,2)/35.;
            (*T)(2,0,2)=M*pow(a,2)*pow(c,2)/35.;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Mutual potential
// ─────────────────────────────────────────────────────────────────────────────

double u_tilde(int dim, int n, mat* t, sp_mat* a, sp_mat* b,
               mat e, cube* TA, cube* TBp){
    // Compute u_tilde_n: the order-n kernel of the mutual potential expansion (Hou 2016).
    // The full potential is U = -G * sum_{n=0}^{N} u_tilde_n / R^(n+1).
    // u_tilde_n = sum_k t_{n,k/2} * [sum over i,j: a(k,i)*b(n-k,j)*e^(i+j)*TA(i')*TBp(j')]
    //
    // k descends from n by 2s (parity of Legendre polynomial — only even/odd harmonics survive).
    // The i1..i5 loops enumerate index k-tuples for body A's contribution;
    // the j1..j5 loops enumerate (n-k)-tuples for body B's contribution.
    // i6 and j6 are determined by the constraint that all indices sum to k and (n-k) respectively.
    mat u((*t).n_cols,1);
    u.zeros();
    for(int k=n;k>-1;k-=2){
        // i1..i6 index the powers of e in body A's terms: e_x^(i1+i4) * e_y^(i2+i5) * e_z^(i3+i6)
        for(int i1=0;i1<k+1;i1++){
            for(int i2=0;i2<k+1-i1;i2++){
                for(int i3=0;i3<k+1-i1-i2;i3++){
                    for(int i4=0;i4<k+1-i1-i2-i3;i4++){
                        for(int i5=0;i5<k+1-i1-i2-i3-i4;i5++){
                            // j1..j6 index the powers of e in body B's terms
                            for(int j1=0;j1<n-k+1;j1++){
                                for(int j2=0;j2<n-k+1-j1;j2++){
                                    for(int j3=0;j3<n-k+1-j1-j2;j3++){
                                        for(int j4=0;j4<n-k+1-j1-j2-j3;j4++){
                                            for(int j5=0;j5<n-k+1-j1-j2-j3-j4;j5++){
                                                int i6=k-i1-i2-i3-i4-i5;   // remainder for A
                                                int j6=n-k-j1-j2-j3-j4-j5; // remainder for B
                                                // Accumulate: a*b * e^(combined powers) * TA * TBp
                                                u(k/2,0)+=(*a)(0,t_ind(k,i1,i2,i3,i4,i5,i6,dim+1))
                                                    *(*b)(0,t_ind(n-k,j1,j2,j3,j4,j5,j6,dim+1))
                                                    *pow(e(0,0),(i1+i4))*pow(e(0,1),(i2+i5))
                                                    *pow(e(0,2),(i3+i6))
                                                    *(*TA)(i1+j1,i2+j2,i3+j3)
                                                    *(*TBp)(i4+j4,i5+j5,i6+j6);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Apply the t_{n,k/2} weight to this k-slice, then sum over all k slices via accu()
        u(k/2,0)=u(k/2,0)*(*t)(n,k/2);
    }
    return accu(u);
}

double du_dx_tilde(int dim, int n, mat* t, sp_mat* a, sp_mat* b,
                   mat e, double R, int dx, cube* TA, cube* TBp){
    // Partial derivative of u_tilde_n with respect to position component x[dx].
    // The only part of u_tilde that depends on x[dx] is the e^(power) factor,
    // since e = r/R and both e and R depend on x.
    // ce is the derivative of e_x^(i1+i4) * e_y^(i2+i5) * e_z^(i3+i6) w.r.t. x[dx],
    // computed via the product rule.  Each of the 8 if/else branches handles a case
    // where one or more of the three exponents is zero (which would cause a negative power).
    mat du((*t).n_cols,1);
    double ce,de_dx0,de_dx1,de_dx2;
    du.zeros();
    // Pre-compute de_i/dx[dx] for each unit-vector component (see de_dx())
    de_dx0=de_dx(e,R,0,dx);
    de_dx1=de_dx(e,R,1,dx);
    de_dx2=de_dx(e,R,2,dx);
    for(int k=n;k>-1;k-=2){
        for(int i1=0;i1<k+1;i1++){
            for(int i2=0;i2<k+1-i1;i2++){
                for(int i3=0;i3<k+1-i1-i2;i3++){
                    for(int i4=0;i4<k+1-i1-i2-i3;i4++){
                        for(int i5=0;i5<k+1-i1-i2-i3-i4;i5++){
                            for(int j1=0;j1<n-k+1;j1++){
                                for(int j2=0;j2<n-k+1-j1;j2++){
                                    for(int j3=0;j3<n-k+1-j1-j2;j3++){
                                        for(int j4=0;j4<n-k+1-j1-j2-j3;j4++){
                                            for(int j5=0;j5<n-k+1-j1-j2-j3-j4;j5++){
                                                int i6=k-i1-i2-i3-i4-i5;
                                                int j6=n-k-j1-j2-j3-j4-j5;
                                                if(i1+i4==0){
                                                    if(i2+i5==0){
                                                        if(i3+i6==0){ ce=0.; }
                                                        else{ ce=(i3+i6)*pow(e(0,0),(i1+i4))*pow(e(0,1),(i2+i5))*pow(e(0,2),(i3+i6-1.))*de_dx2; }
                                                    } else {
                                                        if(i3+i6==0){ ce=(i2+i5)*pow(e(0,0),(i1+i4))*pow(e(0,1),(i2+i5-1.))*pow(e(0,2),(i3+i6))*de_dx1; }
                                                        else{ ce=(i3+i6)*pow(e(0,0),(i1+i4))*pow(e(0,1),(i2+i5))*pow(e(0,2),(i3+i6-1.))*de_dx2+(i2+i5)*pow(e(0,0),(i1+i4))*pow(e(0,1),(i2+i5-1.))*pow(e(0,2),(i3+i6))*de_dx1; }
                                                    }
                                                } else {
                                                    if(i2+i5==0){
                                                        if(i3+i6==0){ ce=(i1+i4)*pow(e(0,0),(i1+i4-1.))*pow(e(0,1),(i2+i5))*pow(e(0,2),(i3+i6))*de_dx0; }
                                                        else{ ce=(i3+i6)*pow(e(0,0),(i1+i4))*pow(e(0,1),(i2+i5))*pow(e(0,2),(i3+i6-1.))*de_dx2+(i1+i4)*pow(e(0,0),(i1+i4-1.))*pow(e(0,1),(i2+i5))*pow(e(0,2),(i3+i6))*de_dx0; }
                                                    } else {
                                                        if(i3+i6==0){ ce=(i1+i4)*pow(e(0,0),(i1+i4-1.))*pow(e(0,1),(i2+i5))*pow(e(0,2),(i3+i6))*de_dx0+(i2+i5)*pow(e(0,0),(i1+i4))*pow(e(0,1),(i2+i5-1.))*pow(e(0,2),(i3+i6))*de_dx1; }
                                                        else{ ce=(i1+i4)*pow(e(0,0),(i1+i4-1.))*pow(e(0,1),(i2+i5))*pow(e(0,2),(i3+i6))*de_dx0+(i2+i5)*pow(e(0,0),(i1+i4))*pow(e(0,1),(i2+i5-1.))*pow(e(0,2),(i3+i6))*de_dx1+(i3+i6)*pow(e(0,0),(i1+i4))*pow(e(0,1),(i2+i5))*pow(e(0,2),(i3+i6-1.))*de_dx2; }
                                                    }
                                                }
                                                du(k/2,0)+=(*a)(0,t_ind(k,i1,i2,i3,i4,i5,i6,dim+1))
                                                    *(*b)(0,t_ind(n-k,j1,j2,j3,j4,j5,j6,dim+1))
                                                    *ce
                                                    *(*TA)(i1+j1,i2+j2,i3+j3)
                                                    *(*TBp)(i4+j4,i5+j5,i6+j6);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        du(k/2,0)=du(k/2,0)*(*t)(n,k/2);
    }
    return accu(du);
}

double de_dx(mat e, double R, int de, int dx){
    // Partial of unit-vector component e[de] = x[de]/R with respect to x[dx].
    // When de==dx:   d(x_i/R)/dx_i = (R^2 - x_i^2)/R^3 = (sum of other x_j^2)/R^3
    //   The loop shifts the index array so ind[] holds the two "other" component indices.
    // When de!=dx:   d(x_de/R)/dx_dx = -x_de * x_dx / R^3  (off-diagonal term)
    mat x=R*e;
    double val;
    if(de==dx){
        int ind[]={0,1,2};
        // Shift ind so that ind[0] and ind[1] hold the two indices != dx
        for(int i=dx;i<2;i++){ ind[i]=ind[i+1]; }
        val=(pow(x(0,ind[0]),2)+pow(x(0,ind[1]),2))/pow(R,3);
    } else {
        val=-x(0,de)*x(0,dx)/pow(R,3);
    }
    return val;
}

void dT_dc(int i, int j, mat C, int q, cube* TA, cube* dT){
    // Partial of the rotated inertia integral T'_{l,m,n} with respect to C(i,j).
    // T' is computed by inertia_rot(), which contains powers of every element of C.
    // Differentiating that expression with respect to C(i,j) picks out only the terms
    // where C(i,j) appears (i.e. row i of C) and reduces its exponent by 1.
    // c holds that derivative of the rotation-matrix power product for a given (i1..j3) tuple.
    // The outer (i,j) arguments identify which element of C we are differentiating against.
    double c;
    (*dT).zeros(q+1,q+1,q+1);
    // Same loop structure as inertia_rot: (l,m,n) are output integral indices,
    // inner 6 loops are the multinomial expansion indices.
    for(int l=0;l<(q+1);l++){
        for(int m=0;m<(q+1-l);m++){
            for(int n=0;n<(q+1-l-m);n++){
                for(double i1=0.;i1<(l+1.);i1++){
                    for(double j1=0.;j1<(l-i1+1.);j1++){
                        for(double i2=0.;i2<(m+1.);i2++){
                            for(double j2=0.;j2<(m-i2+1.);j2++){
                                for(double i3=0.;i3<(n+1.);i3++){
                                    for(double j3=0.;j3<(n-i3+1.);j3++){
                                        if(((i1+i2+i3)<=q)&&((j1+j2+j3)<=q)&&((l+m+n-i1-i2-i3-j1-j2-j3)<=q)){
                                            c=0.;
                                            // Row i=0: C(0,j) contributes to powers i1, j1, (l-i1-j1)
                                            // Row i=1: C(1,j) contributes to powers i2, j2, (m-i2-j2)
                                            // Row i=2: C(2,j) contributes to powers i3, j3, (n-i3-j3)
                                            // The conditionals guard against differentiating a zero exponent.
                                            if(i==0){
                                                if(j==0&&i1>0) c=i1*pow(C(0,0),(i1-1.))*pow(C(0,1),j1)*pow(C(0,2),(l-i1-j1))*pow(C(1,0),i2)*pow(C(1,1),j2)*pow(C(1,2),(m-i2-j2))*pow(C(2,0),i3)*pow(C(2,1),j3)*pow(C(2,2),(n-i3-j3));
                                                else if(j==1&&j1>0) c=j1*pow(C(0,0),i1)*pow(C(0,1),(j1-1.))*pow(C(0,2),(l-i1-j1))*pow(C(1,0),i2)*pow(C(1,1),j2)*pow(C(1,2),(m-i2-j2))*pow(C(2,0),i3)*pow(C(2,1),j3)*pow(C(2,2),(n-i3-j3));
                                                else if(j==2&&(l-j1-i1)>0) c=(l-i1-j1)*pow(C(0,0),i1)*pow(C(0,1),j1)*pow(C(0,2),(l-i1-j1-1.))*pow(C(1,0),i2)*pow(C(1,1),j2)*pow(C(1,2),(m-i2-j2))*pow(C(2,0),i3)*pow(C(2,1),j3)*pow(C(2,2),(n-i3-j3));
                                            } else if(i==1){
                                                if(j==0&&i2>0) c=i2*pow(C(0,0),i1)*pow(C(0,1),j1)*pow(C(0,2),(l-i1-j1))*pow(C(1,0),(i2-1.))*pow(C(1,1),j2)*pow(C(1,2),(m-i2-j2))*pow(C(2,0),i3)*pow(C(2,1),j3)*pow(C(2,2),(n-i3-j3));
                                                else if(j==1&&j2>0) c=j2*pow(C(0,0),i1)*pow(C(0,1),j1)*pow(C(0,2),(l-i1-j1))*pow(C(1,0),i2)*pow(C(1,1),(j2-1.))*pow(C(1,2),(m-i2-j2))*pow(C(2,0),i3)*pow(C(2,1),j3)*pow(C(2,2),(n-i3-j3));
                                                else if(j==2&&(m-i2-j2)>0) c=(m-i2-j2)*pow(C(0,0),i1)*pow(C(0,1),j1)*pow(C(0,2),(l-i1-j1))*pow(C(1,0),i2)*pow(C(1,1),j2)*pow(C(1,2),(m-i2-j2-1.))*pow(C(2,0),i3)*pow(C(2,1),j3)*pow(C(2,2),(n-i3-j3));
                                            } else {
                                                if(j==0&&i3>0) c=i3*pow(C(0,0),i1)*pow(C(0,1),j1)*pow(C(0,2),(l-i1-j1))*pow(C(1,0),i2)*pow(C(1,1),j2)*pow(C(1,2),(m-i2-j2))*pow(C(2,0),(i3-1.))*pow(C(2,1),j3)*pow(C(2,2),(n-i3-j3));
                                                else if(j==1&&j3>0) c=j3*pow(C(0,0),i1)*pow(C(0,1),j1)*pow(C(0,2),(l-i1-j1))*pow(C(1,0),i2)*pow(C(1,1),j2)*pow(C(1,2),(m-i2-j2))*pow(C(2,0),i3)*pow(C(2,1),(j3-1.))*pow(C(2,2),(n-i3-j3));
                                                else if(j==2&&(n-i3-j3)) c=(n-i3-j3)*pow(C(0,0),i1)*pow(C(0,1),j1)*pow(C(0,2),(l-i1-j1))*pow(C(1,0),i2)*pow(C(1,1),j2)*pow(C(1,2),(m-i2-j2))*pow(C(2,0),i3)*pow(C(2,1),j3)*pow(C(2,2),(n-i3-j3-1.));
                                            }
                                            (*dT)(l,m,n)+=(factorial((double)l)/(factorial((double)i1)*factorial((double)j1)*factorial((double)l-i1-j1)))
                                                *(factorial((double)m)/(factorial((double)i2)*factorial((double)j2)*factorial((double)m-i2-j2)))
                                                *(factorial((double)n)/(factorial((double)i3)*factorial((double)j3)*factorial(n-i3-j3)))
                                                *c
                                                *(*TA)(i1+i2+i3,j1+j2+j3,l+m+n-i1-i2-i3-j1-j2-j3);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

double du_dc_tilde(int dim, int n, mat* t, sp_mat* a, sp_mat* b,
                   mat e, cube* TA, cube* dT){
    mat du((*t).n_cols,1);
    du.zeros();
    for(int k=n;k>-1;k-=2){
        for(int i1=0;i1<k+1;i1++){
            for(int i2=0;i2<k+1-i1;i2++){
                for(int i3=0;i3<k+1-i1-i2;i3++){
                    for(int i4=0;i4<k+1-i1-i2-i3;i4++){
                        for(int i5=0;i5<k+1-i1-i2-i3-i4;i5++){
                            for(int j1=0;j1<n-k+1;j1++){
                                for(int j2=0;j2<n-k+1-j1;j2++){
                                    for(int j3=0;j3<n-k+1-j1-j2;j3++){
                                        for(int j4=0;j4<n-k+1-j1-j2-j3;j4++){
                                            for(int j5=0;j5<n-k+1-j1-j2-j3-j4;j5++){
                                                int i6=k-i1-i2-i3-i4-i5;
                                                int j6=n-k-j1-j2-j3-j4-j5;
                                                du(k/2,0)+=(*a)(0,t_ind(k,i1,i2,i3,i4,i5,i6,dim+1))
                                                    *(*b)(0,t_ind(n-k,j1,j2,j3,j4,j5,j6,dim+1))
                                                    *pow(e(0,0),(i1+i4))*pow(e(0,1),(i2+i5))
                                                    *pow(e(0,2),(i3+i6))
                                                    *(*TA)(i1+j1,i2+j2,i3+j3)
                                                    *(*dT)(i4+j4,i5+j5,i6+j6);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        du(k/2,0)=du(k/2,0)*(*t)(n,k/2);
    }
    return accu(du);
}

double du_x(double G, int m, mat* t, sp_mat* a, sp_mat* b,
            mat e, double R, int dx, cube* TA, cube* TBp){
    double du=0.;
    mat x=e*R;
    for(int n=0;n<m+1;n++){
        du+=(-((n+1.)*(x(0,dx)))/pow(R,(n+3.)))*u_tilde(m,n,t,a,b,e,TA,TBp)
            +(1./pow(R,(n+1.)))*du_dx_tilde(m,n,t,a,b,e,R,dx,TA,TBp);
    }
    return -du*G;
}

double du_c(double G, int m, mat* t, sp_mat* a, sp_mat* b,
            mat e, double R, cube* TA, cube* dT){
    double du=0.;
    for(int n=0;n<m+1;n++){
        du+=(1./pow(R,(n+1.)))*du_dc_tilde(m,n,t,a,b,e,TA,dT);
    }
    return -du*G;
}

double potential(double G, int m, mat* t, sp_mat* a, sp_mat* b,
                 mat e, double R, cube* TA, cube* TBp){
    double u=0.;
    for(int n=0;n<m+1;n++){
        u+=(1./pow(R,(n+1.)))*u_tilde(m,n,t,a,b,e,TA,TBp);
    }
    return -u*G;
}

// ─────────────────────────────────────────────────────────────────────────────
// Perturbation functions
// ─────────────────────────────────────────────────────────────────────────────

double kepler(double* n_hyp, double t, double* e_hyp, double* tau_hyp){
    // Solve Kepler's equation for true anomaly theta at time t.
    // M = n*(t - tau) is the mean anomaly.
    // For hyperbolic orbits (e>1): solve M = e*sinh(H) - H for hyperbolic anomaly H,
    //   then convert to true anomaly via the hyperbolic half-angle formula.
    // For elliptic orbits (e<=1): solve M = E - e*sin(E) for eccentric anomaly E,
    //   then convert to true anomaly via the elliptic half-angle formula.
    // Both use Newton-Raphson iteration: x_{n+1} = x_n - f(x_n)/f'(x_n)
    double M=(*n_hyp)*(t-(*tau_hyp));
    double tol=0.001;
    double f,df,theta;
    if((*e_hyp)>1){
        double H=M;
        // Clamp initial guess to the physical range for hyperbolic orbit
        if(abs(H)>acos(-1/(*e_hyp))){
            H=(H>0)?acos(-1/(*e_hyp)):-acos(-1/(*e_hyp));
        }
        f=M-(*e_hyp)*sinh(H)+H;
        // Newton-Raphson: residual f = M - e*sinh(H) + H, derivative df = -e*cosh(H) + 1
        while(abs(f)>tol){
            f=M-(*e_hyp)*sinh(H)+H;
            df=-(*e_hyp)*cosh(H)+1;
            H=H-f/df;
            f=M-(*e_hyp)*sinh(H)+H;
        }
        theta=2*atan(sqrt(((*e_hyp)+1)/((*e_hyp)-1))*tanh(H/2));
    } else {
        double E=M;
        f=E-M-(*e_hyp)*sin(E);
        // Newton-Raphson: residual f = E - M - e*sin(E), derivative df = 1 - e*cos(E)
        while(abs(f)>tol){
            f=E-M-(*e_hyp)*sin(E);
            df=1-(*e_hyp)*cos(E);
            E=E-f/df;
            f=E-M-(*e_hyp)*sin(E);
        }
        theta=2*atan(sqrt((1+(*e_hyp))/(1-(*e_hyp)))*tan(E/2));
    }
    return theta;
}

vec kepler2cart(double* a_hyp, double* e_hyp, double* i_hyp,
                double* RAAN_hyp, double* om_hyp, double f0_hyp,
                double* G, double* Mplanet){
    vec X,Y,Z,X_hyp;
    double mu=(*G)*(*Mplanet);
    X<<1<<endr<<0<<endr<<0<<endr;
    Y<<0<<endr<<1<<endr<<0<<endr;
    Z<<0<<endr<<0<<endr<<1<<endr;
    vec n_omega=cos(*RAAN_hyp)*X+sin(*RAAN_hyp)*Y;
    vec n_perp=-cos(*i_hyp)*sin(*RAAN_hyp)*X+cos(*i_hyp)*cos(*RAAN_hyp)*Y+sin(*i_hyp)*Z;
    vec e_hat=cos(*om_hyp)*n_omega+sin(*om_hyp)*n_perp;
    vec e_perp=-sin(*om_hyp)*n_omega+cos(*om_hyp)*n_perp;
    double p=(*a_hyp)*(1-pow(*e_hyp,2));
    double r_mag=p/(1+(*e_hyp)*cos(f0_hyp));
    vec r=r_mag*(cos(f0_hyp)*e_hat+sin(f0_hyp)*e_perp);
    vec r_hat=r/r_mag;
    vec h_hat=cross(e_hat,e_perp);
    vec r_perp=cross(h_hat,r_hat)/norm(cross(h_hat,r_hat),2);
    double sin_gamma=(*e_hyp)*sin(f0_hyp)/sqrt(1+2*(*e_hyp)*cos(f0_hyp)+pow((*e_hyp),2));
    double cos_gamma=(1+(*e_hyp)*cos(f0_hyp))/sqrt(1+2*(*e_hyp)*cos(f0_hyp)+pow((*e_hyp),2));
    double v_mag=sqrt((mu/p)*(1+2*(*e_hyp)*cos(f0_hyp)+pow((*e_hyp),2)));
    vec v=v_mag*(sin_gamma*r_hat+cos_gamma*r_perp);
    X_hyp<<r(0)<<endr<<r(1)<<endr<<r(2)<<endr<<v(0)<<endr<<v(1)<<endr<<v(2)<<endr;
    return X_hyp;
}

void grav_3BP(vec R_s, mat* NA, mat* pos, double* nu,
              double* G, double* Mplanet, mat* acc_3BP){
    R_s=(*NA).t()*R_s;
    vec R=(*pos);
    (*acc_3BP)=(*G)*(*Mplanet)*((R_s-(1-(*nu)*R))/(pow(norm(R_s-(1-(*nu)*R)),3))
                               -(R_s+(*nu)*R)/(pow(norm(R_s+(*nu)*R),3)));
}

void solar_accel(vec R_s, mat* NA, mat* pos, double* nu,
                 double* G, double* Msun, mat* acc_solar){
    R_s=(*NA).t()*R_s;
    vec R=(*pos);
    (*acc_solar)=(*G)*(*Msun)*((R_s-(1-(*nu)*R))/(pow(norm(R_s-(1-(*nu)*R)),3))
                              -(R_s+(*nu)*R)/(pow(norm(R_s+(*nu)*R),3)));
}

void hill_solar_grav(mat* NA, mat* pos, mat* vel, double* n, mat* acc){
    mat CHp(3,3),CHv(3,3);
    CHp.zeros(); CHv.zeros();
    CHp(0,0)=3.*pow((*n),2.);
    CHv(0,1)=2.*(*n);
    CHv(1,0)=-2.*(*n);
    CHp(2,2)=-pow((*n),2.);
    (*acc)=(*NA).t()*(CHp*(*NA)*(*pos)+CHv*(*NA)*(*vel));
}

void md_tidal_torque(mat* pos, mat* vel, mat* w1, mat* w2,
                     mat* NA, mat* AB, parameters inputs){
    mat rhat=(*NA)*(*pos)/norm(*pos);
    mat vhat=(*NA)*(*vel)/norm(*vel);
    mat rcv=(*NA)*cross(*pos,*vel);
    mat zhat=rcv/norm(rcv);
    mat wsys=rcv/pow(norm(*pos),2.);
    mat phid1=(*NA)*(*w1)-wsys;
    mat phid2=(*NA)*(*w2)-wsys;
    mat gamma_1_vec=phid1-(dot(phid1,rhat))*rhat;
    mat gamma_1_hat=-gamma_1_vec/norm(gamma_1_vec);
    mat gamma_2_vec=phid2-(dot(phid2,rhat))*rhat;
    mat gamma_2_hat=-gamma_2_vec/norm(gamma_2_vec);
    mat yhat=cross(zhat,rhat)/norm(cross(zhat,rhat));
    double g1=3./2.*(*(inputs.love1))*pow(3./(4.*M_PI*(*(inputs.rhoA))),2.)*(*(inputs.G))*pow((*(inputs.TA))(0,0,0),2.)*pow((*(inputs.TB))(0,0,0),2.)*sin(2.*(*(inputs.eps1)))/(*(inputs.refrad1))/pow(norm(*pos),6.);
    double g2=3./2.*(*(inputs.love2))*pow(3./(4.*M_PI*(*(inputs.rhoB))),2.)*(*(inputs.G))*pow((*(inputs.TA))(0,0,0),2.)*pow((*(inputs.TB))(0,0,0),2.)*sin(2.*(*(inputs.eps2)))/(*(inputs.refrad2))/pow(norm(*pos),6.);
    double del1=g1*pow(6./(M_PI*(*(inputs.G))*(*(inputs.rhoA))),.5)/(*(inputs.IA))(0,2);
    double del2=g2*pow(6./(M_PI*(*(inputs.G))*(*(inputs.rhoB))),.5)/(*(inputs.IB))(0,2);
    if(norm(phid1)>abs(del1)){ (*(inputs.tt_1))=g1*(*NA).t()*gamma_1_hat; }
    else{ (*(inputs.tt_1))=norm(phid1)*pow(6./(M_PI*(*(inputs.G))*(*(inputs.rhoA))),-.5)*(*(inputs.IA))(0,2)*(*NA).t()*gamma_1_hat; }
    if(norm(phid2)>abs(del2)){ (*(inputs.tt_2))=g2*(*NA).t()*gamma_2_hat; }
    else{ (*(inputs.tt_2))=norm(phid2)*pow(6./(M_PI*(*(inputs.G))*(*(inputs.rhoB))),-.5)*(*(inputs.IB))(0,2)*(*NA).t()*gamma_2_hat; }
    (*(inputs.tt_orbit))=(1./(*(inputs.m)))*(cross((*(inputs.tt_1)+*(inputs.tt_2)),*pos))/pow(norm(*pos),2.);
}

// ─────────────────────────────────────────────────────────────────────────────
// Equations of motion
// ─────────────────────────────────────────────────────────────────────────────

mat hou_ode(mat x, mat t, parameters inputs){
    // Full F2BP equations of motion from Hou 2016.
    // State vector x is 1×30:
    //   [0:2]  r   — relative position vector (A frame, km)
    //   [3:5]  v   — relative velocity vector (A frame, km/s)
    //   [6:8]  wc  — primary angular velocity (A frame, rad/s)
    //   [9:11] ws  — secondary angular velocity (A frame, rad/s)
    //   [12:20] Cc — inertial-to-A DCM, stored row-major as 9 scalars
    //   [21:29] C  — secondary-to-A DCM, stored row-major as 9 scalars
    // The DCMs are extracted row-major and then transposed so columns = body axes.
    mat r=x.cols(0,2);
    mat v=x.cols(3,5);
    mat wc=x.cols(6,8);
    mat ws=x.cols(9,11);
    mat Cc=x.cols(12,20);
    mat C=x.cols(21,29);
    double time=t(0,0);
    r.reshape(3,1); v.reshape(3,1); wc.reshape(3,1); ws.reshape(3,1);
    Cc.reshape(3,3); Cc=Cc.t();
    C.reshape(3,3);  C=C.t();
    mat Ic_inv(3,3),Is_inv(3,3);
    double R=norm(r,2);
    mat e=r.t()/R;
    Ic_inv<<1./(*(inputs.IA))(0,0)<<0.<<0.<<endr<<0.<<1./(*(inputs.IA))(0,1)<<0.<<endr<<0.<<0.<<1./(*(inputs.IA))(0,2)<<endr;
    Is_inv<<1./(*(inputs.IB))(0,0)<<0.<<0.<<endr<<0.<<1./(*(inputs.IB))(0,1)<<0.<<endr<<0.<<0.<<1./(*(inputs.IB))(0,2)<<endr;
    mat Is_inv_c=C*Is_inv*C.t();
    mat wc_tilde=tilde_op(wc(0,0),wc(1,0),wc(2,0));
    mat ws_s=C.t()*ws;
    mat ws_tilde=tilde_op(ws_s(0,0),ws_s(1,0),ws_s(2,0));
    inertia_rot(C,*(inputs.n),inputs.TB,inputs.TBp);

    mat M_sa(3,1),M_sb(3,1);
    if((*(inputs.flyby_toggle))==1){
        double f0_hyp=kepler(inputs.n_hyp,time,inputs.e_hyp,inputs.tau_hyp);
        vec X_s=kepler2cart(inputs.a_hyp,inputs.e_hyp,inputs.i_hyp,inputs.RAAN_hyp,inputs.om_hyp,f0_hyp,inputs.G,inputs.Mplanet);
        vec R_s; R_s<<X_s(0)<<endr<<X_s(1)<<endr<<X_s(2)<<endr;
        grav_3BP(R_s,&Cc,&r,inputs.nu,inputs.G,inputs.Mplanet,inputs.acc_3BP);
        vec R_sa=Cc.t()*R_s+(*inputs.nu)*(r);
        vec R_sb=Cc.t()*R_s-(1-(*inputs.nu))*(r);
        double Rsa_mag=norm(R_sa,2), Rsb_mag=norm(R_sb,2);
        mat e_sa=R_sa.t()/Rsa_mag, e_sb=R_sb.t()/Rsb_mag;
        mat du_drsa(3,1),du_drsb(3,1);
        du_drsa(0,0)=du_x(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e_sa,Rsa_mag,0,inputs.TA,inputs.TS);
        du_drsa(1,0)=du_x(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e_sa,Rsa_mag,1,inputs.TA,inputs.TS);
        du_drsa(2,0)=du_x(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e_sa,Rsa_mag,2,inputs.TA,inputs.TS);
        du_drsb(0,0)=du_x(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e_sb,Rsb_mag,0,inputs.TBp,inputs.TS);
        du_drsb(1,0)=du_x(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e_sb,Rsb_mag,1,inputs.TBp,inputs.TS);
        du_drsb(2,0)=du_x(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e_sb,Rsb_mag,2,inputs.TBp,inputs.TS);
        M_sa=cross(R_sa,du_drsa);
        M_sb=cross(R_sb,du_drsb);
    } else {
        (*(inputs.acc_3BP)).zeros();
        M_sa.zeros(); M_sb.zeros();
    }

    mat M_suna(3,1),M_sunb(3,1);
    if((*(inputs.helio_toggle))==1){
        double f0_helio=kepler(inputs.n_helio,time,inputs.e_helio,inputs.tau_helio);
        vec X_sun=kepler2cart(inputs.a_helio,inputs.e_helio,inputs.i_helio,inputs.RAAN_helio,inputs.om_helio,f0_helio,inputs.G,inputs.Msolar);
        vec R_sun; R_sun<<X_sun(0)<<endr<<X_sun(1)<<endr<<X_sun(2)<<endr;
        solar_accel(R_sun,&Cc,&r,inputs.nu,inputs.G,inputs.Msolar,inputs.acc_solar);
        vec R_suna=Cc.t()*R_sun+(*inputs.nu)*(r);
        vec R_sunb=Cc.t()*R_sun-(1-(*inputs.nu))*(r);
        double Rsuna_mag=norm(R_suna,2), Rsunb_mag=norm(R_sunb,2);
        mat e_suna=R_suna.t()/Rsuna_mag, e_sunb=R_sunb.t()/Rsunb_mag;
        mat du_drsuna(3,1),du_drsunb(3,1);
        du_drsuna(0,0)=du_x(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e_suna,Rsuna_mag,0,inputs.TA,inputs.Tsun);
        du_drsuna(1,0)=du_x(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e_suna,Rsuna_mag,1,inputs.TA,inputs.Tsun);
        du_drsuna(2,0)=du_x(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e_suna,Rsuna_mag,2,inputs.TA,inputs.Tsun);
        du_drsunb(0,0)=du_x(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e_sunb,Rsunb_mag,0,inputs.TBp,inputs.Tsun);
        du_drsunb(1,0)=du_x(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e_sunb,Rsunb_mag,1,inputs.TBp,inputs.Tsun);
        du_drsunb(2,0)=du_x(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e_sunb,Rsunb_mag,2,inputs.TBp,inputs.Tsun);
        M_suna=cross(R_suna,du_drsuna);
        M_sunb=cross(R_sunb,du_drsunb);
    } else {
        (*(inputs.acc_solar)).zeros();
        M_suna.zeros(); M_sunb.zeros();
    }

    if((*(inputs.sg_toggle))==1){
        hill_solar_grav(&Cc,&r,&v,inputs.mean_motion,inputs.sg_acc);
    } else {
        (*(inputs.sg_acc)).zeros();
    }

    if(*(inputs.tt_toggle)==1){
        md_tidal_torque(&r,&v,&wc,&ws,&Cc,&C,inputs);
    } else {
        (*(inputs.tt_1)).zeros();
        (*(inputs.tt_2)).zeros();
        (*(inputs.tt_orbit)).zeros();
    }

    dT_dc(0,0,C,*(inputs.n),inputs.TB,&((*(inputs.dT))(0,0)));
    dT_dc(0,1,C,*(inputs.n),inputs.TB,&((*(inputs.dT))(0,1)));
    dT_dc(0,2,C,*(inputs.n),inputs.TB,&((*(inputs.dT))(0,2)));
    dT_dc(1,0,C,*(inputs.n),inputs.TB,&((*(inputs.dT))(1,0)));
    dT_dc(1,1,C,*(inputs.n),inputs.TB,&((*(inputs.dT))(1,1)));
    dT_dc(1,2,C,*(inputs.n),inputs.TB,&((*(inputs.dT))(1,2)));
    dT_dc(2,0,C,*(inputs.n),inputs.TB,&((*(inputs.dT))(2,0)));
    dT_dc(2,1,C,*(inputs.n),inputs.TB,&((*(inputs.dT))(2,1)));
    dT_dc(2,2,C,*(inputs.n),inputs.TB,&((*(inputs.dT))(2,2)));

    mat du_dr(3,1),du_dalpha(3,1),du_dbeta(3,1),du_dgam(3,1);
    du_dr(0,0)=du_x(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e,R,0,inputs.TA,inputs.TBp);
    du_dr(1,0)=du_x(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e,R,1,inputs.TA,inputs.TBp);
    du_dr(2,0)=du_x(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e,R,2,inputs.TA,inputs.TBp);
    du_dalpha(0,0)=du_c(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e,R,inputs.TA,&((*(inputs.dT))(0,0)));
    du_dalpha(1,0)=du_c(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e,R,inputs.TA,&((*(inputs.dT))(1,0)));
    du_dalpha(2,0)=du_c(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e,R,inputs.TA,&((*(inputs.dT))(2,0)));
    du_dbeta(0,0)=du_c(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e,R,inputs.TA,&((*(inputs.dT))(0,1)));
    du_dbeta(1,0)=du_c(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e,R,inputs.TA,&((*(inputs.dT))(1,1)));
    du_dbeta(2,0)=du_c(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e,R,inputs.TA,&((*(inputs.dT))(2,1)));
    du_dgam(0,0)=du_c(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e,R,inputs.TA,&((*(inputs.dT))(0,2)));
    du_dgam(1,0)=du_c(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e,R,inputs.TA,&((*(inputs.dT))(1,2)));
    du_dgam(2,0)=du_c(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e,R,inputs.TA,&((*(inputs.dT))(2,2)));

    mat La=diagmat(*(inputs.IA))*wc;
    mat Lb=C*diagmat(*(inputs.IB))*ws_s;
    mat Mb=-cross(C.col(0),du_dalpha)-cross(C.col(1),du_dbeta)-cross(C.col(2),du_dgam);
    mat Ma=cross(r,du_dr)-Mb;
    mat rd=cross(r,wc)+v;
    mat vd=cross(v,wc)-(1./(*(inputs.m)))*du_dr+(*(inputs.sg_acc))+(*(inputs.acc_3BP))+(*(inputs.acc_solar))+(*(inputs.tt_orbit));
    mat Cdc=Cc*wc_tilde;
    mat Cd=C*ws_tilde-wc_tilde*C;
    mat CdT=-ws_tilde*C.t()+C.t()*wc_tilde;
    mat wcd=Ic_inv*(cross(La,wc)+Ma-(*(inputs.tt_1)));
    mat wsd=Is_inv_c*(cross(Lb,wc)+Mb-Cd*diagmat(*(inputs.IB))*C.t()*ws-C*diagmat(*(inputs.IB))*CdT*ws-(*(inputs.tt_2)));

    mat xd(30,1);
    xd<<rd(0,0)<<endr<<rd(1,0)<<endr<<rd(2,0)<<endr
      <<vd(0,0)<<endr<<vd(1,0)<<endr<<vd(2,0)<<endr
      <<wcd(0,0)<<endr<<wcd(1,0)<<endr<<wcd(2,0)<<endr
      <<wsd(0,0)<<endr<<wsd(1,0)<<endr<<wsd(2,0)<<endr
      <<Cdc(0,0)<<endr<<Cdc(0,1)<<endr<<Cdc(0,2)<<endr
      <<Cdc(1,0)<<endr<<Cdc(1,1)<<endr<<Cdc(1,2)<<endr
      <<Cdc(2,0)<<endr<<Cdc(2,1)<<endr<<Cdc(2,2)<<endr
      <<Cd(0,0)<<endr<<Cd(0,1)<<endr<<Cd(0,2)<<endr
      <<Cd(1,0)<<endr<<Cd(1,1)<<endr<<Cd(1,2)<<endr
      <<Cd(2,0)<<endr<<Cd(2,1)<<endr<<Cd(2,2)<<endr;
    return xd;
}

// ─────────────────────────────────────────────────────────────────────────────
// LGVI support
// ─────────────────────────────────────────────────────────────────────────────

void map_potential_partials(mat* C, mat* r, parameters inputs,
                            mat* du_dr, mat* M){
    double R=norm(*r,2);
    mat e=(*r).t()/R;
    mat du_dalpha(3,1),du_dbeta(3,1),du_dgam(3,1);
    inertia_rot(*C,*(inputs.n),inputs.TB,inputs.TBp);
    dT_dc(0,0,*C,*(inputs.n),inputs.TB,&((*(inputs.dT))(0,0)));
    dT_dc(0,1,*C,*(inputs.n),inputs.TB,&((*(inputs.dT))(0,1)));
    dT_dc(0,2,*C,*(inputs.n),inputs.TB,&((*(inputs.dT))(0,2)));
    dT_dc(1,0,*C,*(inputs.n),inputs.TB,&((*(inputs.dT))(1,0)));
    dT_dc(1,1,*C,*(inputs.n),inputs.TB,&((*(inputs.dT))(1,1)));
    dT_dc(1,2,*C,*(inputs.n),inputs.TB,&((*(inputs.dT))(1,2)));
    dT_dc(2,0,*C,*(inputs.n),inputs.TB,&((*(inputs.dT))(2,0)));
    dT_dc(2,1,*C,*(inputs.n),inputs.TB,&((*(inputs.dT))(2,1)));
    dT_dc(2,2,*C,*(inputs.n),inputs.TB,&((*(inputs.dT))(2,2)));
    (*du_dr)(0,0)=du_x(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e,R,0,inputs.TA,inputs.TBp);
    (*du_dr)(1,0)=du_x(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e,R,1,inputs.TA,inputs.TBp);
    (*du_dr)(2,0)=du_x(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e,R,2,inputs.TA,inputs.TBp);
    du_dalpha(0,0)=du_c(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e,R,inputs.TA,&((*(inputs.dT))(0,0)));
    du_dalpha(1,0)=du_c(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e,R,inputs.TA,&((*(inputs.dT))(1,0)));
    du_dalpha(2,0)=du_c(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e,R,inputs.TA,&((*(inputs.dT))(2,0)));
    du_dbeta(0,0)=du_c(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e,R,inputs.TA,&((*(inputs.dT))(0,1)));
    du_dbeta(1,0)=du_c(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e,R,inputs.TA,&((*(inputs.dT))(1,1)));
    du_dbeta(2,0)=du_c(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e,R,inputs.TA,&((*(inputs.dT))(2,1)));
    du_dgam(0,0)=du_c(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e,R,inputs.TA,&((*(inputs.dT))(0,2)));
    du_dgam(1,0)=du_c(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e,R,inputs.TA,&((*(inputs.dT))(1,2)));
    du_dgam(2,0)=du_c(*(inputs.G),*(inputs.n),inputs.tk,inputs.a,inputs.b,e,R,inputs.TA,&((*(inputs.dT))(2,2)));
    (*M)=cross((*C).col(0),du_dalpha)+cross((*C).col(1),du_dbeta)+cross((*C).col(2),du_dgam);
}

void F_cayley_calc(double h, mat* g, mat* I, mat* f,
                   mat* F, mat* G, mat* grad_G){
    // Solve for the rotation vector f used in the LGVI integrator via the Cayley map.
    // The Cayley map relates a 3-vector f to a rotation matrix F = (I+f~)(I-f~)^{-1}.
    // We need f satisfying: g + g×f + (g·f)f - 2*I*f = 0  (nonlinear equation in f).
    // Newton-Raphson iteration: f_{n+1} = f_n - grad_G^{-1} * G(f_n)
    // g is scaled by h before the iteration so f is dimensionless.
    // If Newton fails (> 100 iterations), fall back to the exponential map solver.
    (*f).fill(.0001);
    (*G)=ones(3,1);
    (*g)=h*(*g);
    int check=0;
    while(norm(*G)>(1.e-15)){
        (*G)=(*g)+cross(*g,*f)+dot(*g,*f)*(*f)-2.*(*I)*(*f);
        (*grad_G)=tilde_opv(*g)+dot(*g,*f)*eye(3,3)+(*f)*(*g).t()-2.*(*I);
        (*f)=(*f)+inv(*grad_G)*(- (*G));
        check++;
        if(check>100){
            // Cayley map Newton didn't converge; try exponential map
            double cay_err=norm(*G);
            mat cay_f=(*f);
            F_exp_calc(h,g,I,f,F,G,grad_G);
            if(norm((*g)-(*G))>cay_err){ (*f)=cay_f; }
            break;
        }
    }
    if(check<=100){
        // Converged: assemble the rotation matrix from f via the Cayley formula
        (*F)=(eye(3,3)+tilde_opv(*f))*inv(eye(3,3)-tilde_opv(*f));
    }
}

void F_exp_calc(double h, mat* g, mat* I, mat* f,
                mat* F, mat* G, mat* grad_G){
    // Fallback rotation solver using the exponential (Rodrigues) map.
    // Finds f (rotation axis × angle) such that the exponential map gives rotation matrix F.
    // The equation to solve is: G(f) = sin(|f|)*I*f/|f| + (1-cos|f|)*f×(I*f)/|f|^2 = g
    // grad_G is the Jacobian of G w.r.t. f (analytic expression).
    // Newton-Raphson: f_{n+1} = f_n + grad_G^{-1} * (g - G(f_n))
    // F is then assembled from f via Rodrigues' formula: F = I + sin|f|*f~ + (1-cos|f|)*f~^2
    int check=0;
    (*f).fill(.1);
    (*G)=ones(3,1);
    while(norm((*g)-(*G))>(1.e-15)){
        (*G)=sin(norm(*f))*(*I)*(*f)/norm(*f)+(1-cos(norm(*f)))*cross(*f,(*I)*(*f))/pow(norm(*f),2);
        (*grad_G)=(norm(*f)*cos(norm(*f))-sin(norm(*f)))*(*I)*(*f)*(*f).t()/pow(norm(*f),3)
                  +sin(norm(*f))*(*I)/norm(*f)
                  +(norm(*f)*sin(norm(*f))-2.*(1.-cos(norm(*f))))*cross(*f,(*I)*(*f))*(*f).t()/pow(norm(*f),4)
                  +(1.-cos(norm(*f)))*(-tilde_opv((*I)*(*f))+tilde_opv(*f)*(*I))/pow(norm(*f),2);
        (*f)=(*f)+inv(*grad_G)*((*g)-(*G));
        check++;
        if(check>100){ break; }
    }
    // Rodrigues' formula for the rotation matrix given rotation vector f
    (*F)=eye(3,3)+sin(norm(*f))*tilde_opv(*f)/norm(*f)+(1.-cos(norm(*f)))*tilde_opv(*f)*tilde_opv(*f)/pow(norm(*f),2);
}

void hamiltonian_map(double h, parameters inputs, mat* x, mat* x_out,
                     mat* fab, mat* fna, mat* Fab, mat* Fna,
                     mat* g_map, mat* G_map, mat* grad_G,
                     mat* du_dr_n, mat* M_n, mat* x0){
    double nr=norm((*x0).cols(0,2));
    double nm=(*(inputs.m));
    double nt=sqrt(*(inputs.G)*nm/pow(nr,3));
    h=h*nt;
    mat r=(*x).cols(0,2)/nr;
    mat v=(*x).cols(3,5)/nr/nt;
    mat wc=(*x).cols(6,8)/nt;
    mat ws=(*x).cols(9,11)/nt;
    mat Cc=(*x).cols(12,20);
    mat C=(*x).cols(21,29);
    r.reshape(3,1); v.reshape(3,1); wc.reshape(3,1); ws.reshape(3,1);
    Cc.reshape(3,3); Cc=Cc.t();
    C.reshape(3,3);  C=C.t();
    mat ws_s=C.t()*ws;
    mat du_dr_n1(3,1),M_n1(3,1);
    mat IBr=C*diagmat(*(inputs.IB))*C.t()/(nm*pow(nr,2.));
    (*g_map)=C*diagmat(*(inputs.IB))*C.t()*ws/(nm*pow(nr,2.))-(h/2.)*(*M_n)/(nm*pow(nr,2.)*pow(nt,2.));
    mat IdB_rot=C*(*(inputs.IdB))*C.t()/(nm*pow(nr,2.));
    F_cayley_calc(h,g_map,&IBr,fab,Fab,G_map,grad_G);
    (*g_map)=diagmat(*(inputs.IA))*wc/(nm*pow(nr,2.))+(h/2.)*cross(r,(*du_dr_n)/(nm*nr*pow(nt,2.)))+(h/2.)*(*M_n)/(nm*pow(nr,2.)*pow(nt,2.));
    mat IH=diagmat(*(inputs.IA))/(nm*pow(nr,2.));
    F_cayley_calc(h,g_map,&IH,fna,Fna,G_map,grad_G);
    mat r_n1=(*Fna).t()*(r+h*v-nm*(pow(h,2.)/(2.*(*(inputs.m))))*(*du_dr_n)/(nr*nm*pow(nt,2.)));
    mat C_n1=(*Fna).t()*(*Fab)*C;
    mat r_n1n=r_n1*nr;
    map_potential_partials(&C_n1,&r_n1n,inputs,&du_dr_n1,&M_n1);
    mat v_n1=(*Fna).t()*(v-nm*(h/(2.*(*(inputs.m))))*(*du_dr_n)/(nr*nm*pow(nt,2.)))-nm*(h/(2.*(*inputs.m)))*du_dr_n1/(nr*nm*pow(nt,2.));
    mat wc_n1=inv(diagmat(*(inputs.IA))/(nm*pow(nr,2.)))*((*Fna).t()*(diagmat(*(inputs.IA))/(nm*pow(nr,2.))*wc+(h/2.)*cross(r,(*du_dr_n)/(nr*nm*pow(nt,2.)))+(h/2.)*(*M_n)/(nm*pow(nr,2.)*pow(nt,2.)))+(h/2.)*cross(r_n1,du_dr_n1/(nr*nm*pow(nt,2.)))+(h/2.)*M_n1/(nm*pow(nr,2.)*pow(nt,2.)));
    mat ws_n1=C_n1*inv(diagmat(*(inputs.IB))/(nm*pow(nr,2.)))*C_n1.t()*((*Fna).t()*(C*(diagmat(*(inputs.IB))/(nm*pow(nr,2.)))*C.t()*ws-(h/2.)*(*M_n)/(nm*pow(nr,2.)*pow(nt,2.)))-(h/2.)*M_n1/(nm*pow(nr,2.)*pow(nt,2.)));
    mat Cc_n1=Cc*(*Fna);
    r_n1=r_n1n;
    v_n1=v_n1*nr*nt;
    ws_n1=ws_n1*nt;
    wc_n1=wc_n1*nt;
    (*x_out)<<r_n1(0,0)<<r_n1(1,0)<<r_n1(2,0)
            <<v_n1(0,0)<<v_n1(1,0)<<v_n1(2,0)
            <<wc_n1(0,0)<<wc_n1(1,0)<<wc_n1(2,0)
            <<ws_n1(0,0)<<ws_n1(1,0)<<ws_n1(2,0)
            <<Cc_n1(0,0)<<Cc_n1(0,1)<<Cc_n1(0,2)
            <<Cc_n1(1,0)<<Cc_n1(1,1)<<Cc_n1(1,2)
            <<Cc_n1(2,0)<<Cc_n1(2,1)<<Cc_n1(2,2)
            <<C_n1(0,0)<<C_n1(0,1)<<C_n1(0,2)
            <<C_n1(1,0)<<C_n1(1,1)<<C_n1(1,2)
            <<C_n1(2,0)<<C_n1(2,1)<<C_n1(2,2)<<endr;
    (*du_dr_n)=du_dr_n1;
    (*M_n)=M_n1;
}

// ─────────────────────────────────────────────────────────────────────────────
// Library integrators (no file I/O, vector output, out_freq decimation)
// ─────────────────────────────────────────────────────────────────────────────

static void push_state(double t_val, const mat& y,
                       const vec& X_s, const vec& X_helio,
                       int flyby_tog, int helio_tog,
                       parameters& inp,
                       std::vector<double>& times_out,
                       std::vector<double>& states_out,
                       std::vector<double>& hyp_out,
                       std::vector<double>& solar_out,
                       std::vector<double>& potential_out){
    times_out.push_back(t_val);
    for(int c=0;c<30;c++) states_out.push_back(y(0,c));
    if(flyby_tog)  for(int i=0;i<6;i++) hyp_out.push_back(X_s(i));
    if(helio_tog)  for(int i=0;i<6;i++) solar_out.push_back(X_helio(i));

    // Compute and store mutual gravitational potential at this output step.
    // Mirror hou_ode's extraction exactly: reshape to column vector, then
    // form e as the *transposed* (row) unit vector — potential() expects 1×3.
    mat r_vec = y.cols(0, 2); r_vec.reshape(3, 1);  // (3,1) position in A frame
    double R  = norm(r_vec, 2);
    mat e     = r_vec.t() / R;                       // (1,3) unit direction — matches hou_ode
    mat C     = y.cols(21, 29); C.reshape(3, 3); C = C.t();  // B→A rotation matrix
    inertia_rot(C, *(inp.n), inp.TB, inp.TBp);
    double U  = potential(*(inp.G), *(inp.n), inp.tk, inp.a, inp.b,
                          e, R, inp.TA, inp.TBp);
    potential_out.push_back(U);
}

// RK4 fixed-step
static void rk4_lib(double t0, double tf, mat x0, double h, double out_freq,
                    parameters inp,
                    std::vector<double>& times_out,
                    std::vector<double>& states_out,
                    std::vector<double>& hyp_out,
                    std::vector<double>& solar_out,
                    std::vector<double>& potential_out){
    double f0_hyp=kepler(inp.n_hyp,t0,inp.e_hyp,inp.tau_hyp);
    vec X_s=kepler2cart(inp.a_hyp,inp.e_hyp,inp.i_hyp,inp.RAAN_hyp,inp.om_hyp,f0_hyp,inp.G,inp.Mplanet);
    double f0_helio=kepler(inp.n_helio,t0,inp.e_helio,inp.tau_helio);
    vec X_helio=kepler2cart(inp.a_helio,inp.e_helio,inp.i_helio,inp.RAAN_helio,inp.om_helio,f0_helio,inp.G,inp.Msolar);

    bool decimate=(out_freq>0.0);
    double t_next=t0;
    mat y=x0;
    double t=t0;

    push_state(t,y,X_s,X_helio,*inp.flyby_toggle,*inp.helio_toggle,
               inp,times_out,states_out,hyp_out,solar_out,potential_out);
    if(decimate) t_next+=out_freq;

    while(t<tf){
        double dt=h;
        if(t+dt>tf) dt=tf-t;  // clamp last step to hit tf exactly

        // RK4 stage evaluations at t, t+dt/2 (twice), and t+dt
        mat tm(1,1); tm(0,0)=t;
        mat th2(1,1); th2(0,0)=t+dt*0.5;
        mat th(1,1);  th(0,0)=t+dt;

        mat k1=hou_ode(y,             tm, inp);
        mat k2=hou_ode(y+(dt/2.)*k1.t(),th2,inp);
        mat k3=hou_ode(y+(dt/2.)*k2.t(),th2,inp);
        mat k4=hou_ode(y+dt*k3.t(),   th, inp);
        y=y+(dt/6.)*(k1.t()+2.*k2.t()+2.*k3.t()+k4.t());
        t+=dt;

        if(*inp.flyby_toggle){
            f0_hyp=kepler(inp.n_hyp,t,inp.e_hyp,inp.tau_hyp);
            X_s=kepler2cart(inp.a_hyp,inp.e_hyp,inp.i_hyp,inp.RAAN_hyp,inp.om_hyp,f0_hyp,inp.G,inp.Mplanet);
        }
        if(*inp.helio_toggle){
            f0_helio=kepler(inp.n_helio,t,inp.e_helio,inp.tau_helio);
            X_helio=kepler2cart(inp.a_helio,inp.e_helio,inp.i_helio,inp.RAAN_helio,inp.om_helio,f0_helio,inp.G,inp.Msolar);
        }
        if(!decimate||t>=t_next-1e-9){
            push_state(t,y,X_s,X_helio,*inp.flyby_toggle,*inp.helio_toggle,
                       inp,times_out,states_out,hyp_out,solar_out,potential_out);
            if(decimate) t_next+=out_freq;
        }
    }
}

// RK7(8) Dormand-Prince adaptive
static void rk87_lib(double t0, double tf, mat x0, double rel_tol, double out_freq,
                     parameters inp,
                     std::vector<double>& times_out,
                     std::vector<double>& states_out,
                     std::vector<double>& hyp_out,
                     std::vector<double>& solar_out,
                     std::vector<double>& potential_out){
    mat c_i,a_i_j,b_8,b_7;
    c_i<<1./18.<<endr<<1./12.<<endr<<1./8.<<endr<<5./16.<<endr
       <<3./8.<<endr<<59./400.<<endr<<93./200.<<endr<<5490023248./9719169821.<<endr
       <<13./20.<<endr<<1201146811./1299019798.<<endr<<1.<<endr<<1.<<endr;

    a_i_j<<1./18.<<1./48.<<1./32.<<5./16.<<3./80.<<29443841./614563906.<<16016141./946692911.<<39632708./573591083.<<246121993./1340847787.<<-1028468189./846180014.<<185892177./718116043.<<403863854./491063109.<<endr
         <<0.<<1./16.<<0.<<0.<<0.<<0.<<0.<<0.<<0.<<0.<<0.<<0.<<endr
         <<0.<<0.<<3./32.<<-75./64.<<0.<<0.<<0.<<0.<<0.<<0.<<0.<<0.<<endr
         <<0.<<0.<<0.<<75./64.<<3./16.<<77736538./692538347.<<61564180./158732637.<<-433636366./683701615.<<-37695042795./15268766246.<<8478235783./508512852.<<-3185094517./667107341.<<-5068492393./434740067.<<endr
         <<0.<<0.<<0.<<0.<<3./20.<<-28693883./1125000000.<<22789713./633445777.<<-421739975./2616292301.<<-309121744./1061227803.<<1311729495./1432422823.<<-477755414./1098053517.<<-411421997./543043805.<<endr
         <<0.<<0.<<0.<<0.<<0.<<23124283./1800000000.<<545815736./2771057229.<<100302831./723423059.<<-12992083./490766935.<<-10304129995./1701304382.<<-703635378./230739211.<<652783627./914296604.<<endr
         <<0.<<0.<<0.<<0.<<0.<<0.<<-180193667./1043307555.<<790204164./839813087.<<6005943493./2108947869.<<-48777925059./3047939560.<<5731566787./1027545527.<<11173962825./925320556.<<endr
         <<0.<<0.<<0.<<0.<<0.<<0.<<0.<<800635310./3783071287.<<393006217./1396673457.<<15336726248./1032824649.<<5232866602./850066563.<<-13158990841./6184727034.<<endr
         <<0.<<0.<<0.<<0.<<0.<<0.<<0.<<0.<<123872331./1001029789.<<-45442868181./3398467696.<<-4093664535./808688257.<<3936647629./1978049680.<<endr
         <<0.<<0.<<0.<<0.<<0.<<0.<<0.<<0.<<0.<<3065993473./597172653.<<3962137247./1805957418.<<-160528059./685178525.<<endr
         <<0.<<0.<<0.<<0.<<0.<<0.<<0.<<0.<<0.<<0.<<65686358./487910083.<<248638103./1413531060.<<endr
         <<0.<<0.<<0.<<0.<<0.<<0.<<0.<<0.<<0.<<0.<<0.<<0.<<endr
         <<0.<<0.<<0.<<0.<<0.<<0.<<0.<<0.<<0.<<0.<<0.<<0.<<endr;

    b_8<<14005451./335480064.<<endr<<0.<<endr<<0.<<endr<<0.<<endr<<0.<<endr
       <<-59238493./1068277825.<<endr<<181606767./758867731.<<endr<<561292985./797845732.<<endr
       <<-1041891430./1371343529.<<endr<<760417239./1151165299.<<endr
       <<118820643./751138087.<<endr<<-528747749./2220607170.<<endr<<1./4.<<endr;

    b_7<<13451932./455176623.<<endr<<0.<<endr<<0.<<endr<<0.<<endr<<0.<<endr
       <<-808719846./976000145.<<endr<<1757004468./5645159321.<<endr<<656045339./265891186.<<endr
       <<-3867574721./1518517206.<<endr<<465885868./322736535.<<endr
       <<53011238./667516719.<<endr<<2./45.<<endr<<0.<<endr;

    double powv=1./8.;
    double hmax=(tf-t0)/2.5;
    double h=(tf-t0)/50.;
    if(h>.1) h=.1;
    if(h>hmax) h=hmax;
    double eps=5.e-16;

    double f0_hyp=kepler(inp.n_hyp,t0,inp.e_hyp,inp.tau_hyp);
    vec X_s=kepler2cart(inp.a_hyp,inp.e_hyp,inp.i_hyp,inp.RAAN_hyp,inp.om_hyp,f0_hyp,inp.G,inp.Mplanet);
    double f0_helio=kepler(inp.n_helio,t0,inp.e_helio,inp.tau_helio);
    vec X_helio=kepler2cart(inp.a_helio,inp.e_helio,inp.i_helio,inp.RAAN_helio,inp.om_helio,f0_helio,inp.G,inp.Msolar);

    bool decimate=(out_freq>0.0);
    double t_next=t0;

    mat x=x0;
    double t=t0;
    mat f; f.zeros(x.n_cols,13);

    push_state(t,x,X_s,X_helio,*inp.flyby_toggle,*inp.helio_toggle,
               inp,times_out,states_out,hyp_out,solar_out,potential_out);
    if(decimate) t_next+=out_freq;

    while(t<tf){
        if(t+h>tf) h=tf-t;  // clamp step to hit tf exactly
        mat tm(1,1); tm(0,0)=t;
        // Stage 0: derivative at current state
        f.col(0)=hou_ode(x,tm,inp);
        // Stages 1-12: each stage uses previous stage derivatives weighted by Butcher tableau columns
        for(int j=1;j<13;j++){
            mat tc(1,1); tc(0,0)=t+c_i(j-1,0)*h;
            f.col(j)=hou_ode(x+h*trans(f*a_i_j.col(j-1)),tc,inp);
        }
        // 8th-order and 7th-order solutions; error estimate is their difference
        mat sol1=x+h*trans(f*b_8);
        mat sol2=x+h*trans(f*b_7);
        double err=abs(norm(sol1-sol2,"inf"));
        double tau=rel_tol*norm(x,"inf");
        if(err<=tau){
            // Accept step: advance using the 7th-order solution
            t+=h; x=sol2;
            if(*inp.flyby_toggle){
                f0_hyp=kepler(inp.n_hyp,t,inp.e_hyp,inp.tau_hyp);
                X_s=kepler2cart(inp.a_hyp,inp.e_hyp,inp.i_hyp,inp.RAAN_hyp,inp.om_hyp,f0_hyp,inp.G,inp.Mplanet);
            }
            if(*inp.helio_toggle){
                f0_helio=kepler(inp.n_helio,t,inp.e_helio,inp.tau_helio);
                X_helio=kepler2cart(inp.a_helio,inp.e_helio,inp.i_helio,inp.RAAN_helio,inp.om_helio,f0_helio,inp.G,inp.Msolar);
            }
            if(!decimate||t>=t_next-1e-9){
                push_state(t,x,X_s,X_helio,*inp.flyby_toggle,*inp.helio_toggle,
                           inp,times_out,states_out,hyp_out,solar_out,potential_out);
                if(decimate) t_next+=out_freq;
            }
        }
        if(err==0.) err=10.*eps;  // avoid divide-by-zero in step size update
        // Standard adaptive step size formula: h_new = 0.9 * h * (tau/err)^(1/8)
        h=fmin(hmax,0.9*h*pow(tau/err,powv));
        if(abs(h)<=eps) break;  // step size collapsed — stop
    }
}

// Adams-Bashforth-Moulton 4th-order predictor-corrector
static void ABM_lib(double t0, double tf, mat x0, double h, double out_freq,
                    parameters inp,
                    std::vector<double>& times_out,
                    std::vector<double>& states_out,
                    std::vector<double>& hyp_out,
                    std::vector<double>& solar_out,
                    std::vector<double>& potential_out){
    double f0_hyp=kepler(inp.n_hyp,t0,inp.e_hyp,inp.tau_hyp);
    vec X_s=kepler2cart(inp.a_hyp,inp.e_hyp,inp.i_hyp,inp.RAAN_hyp,inp.om_hyp,f0_hyp,inp.G,inp.Mplanet);
    double f0_helio=kepler(inp.n_helio,t0,inp.e_helio,inp.tau_helio);
    vec X_helio=kepler2cart(inp.a_helio,inp.e_helio,inp.i_helio,inp.RAAN_helio,inp.om_helio,f0_helio,inp.G,inp.Msolar);

    bool decimate=(out_freq>0.0);
    double t_next=t0;
    mat y=x0;
    double t=t0;

    push_state(t,y,X_s,X_helio,*inp.flyby_toggle,*inp.helio_toggle,
               inp,times_out,states_out,hyp_out,solar_out,potential_out);
    if(decimate) t_next+=out_freq;

    // derivative history: f1=prev, f2=2prev, f3=3prev (populated during startup)
    mat f0,f1,f2,f3;
    int startup=0; // number of completed RK4 startup steps

    while(t<tf){
        double dt=h;
        if(t+dt>tf) dt=tf-t;
        mat tm(1,1);  tm(0,0)=t;
        mat th2(1,1); th2(0,0)=t+dt*0.5;
        mat th(1,1);  th(0,0)=t+dt;
        mat y_new;

        if(startup<3){
            // RK4 startup — ABM needs 4 derivative history points (f0..f3).
            // We use 3 RK4 steps to populate f1, f2, f3 before switching to ABM.
            mat k1=hou_ode(y,             tm, inp);
            mat k2=hou_ode(y+(dt/2.)*k1.t(),th2,inp);
            mat k3=hou_ode(y+(dt/2.)*k2.t(),th2,inp);
            mat k4=hou_ode(y+dt*k3.t(),   th, inp);
            y_new=y+(dt/6.)*(k1.t()+2.*k2.t()+2.*k3.t()+k4.t());
            // Store derivative at the start of this step into the rolling history.
            // After 3 startup steps: f3=deriv at t0, f2=deriv at t0+h, f1=deriv at t0+2h
            if(startup==0) f3=k1;
            else if(startup==1) f2=k1;
            else f1=k1;
            startup++;
        } else {
            // Adams-Bashforth 4th-order predictor, then Adams-Moulton 4th-order corrector.
            // Predictor uses the last 4 derivatives (f0=current, f1=prev, f2=2prev, f3=3prev).
            f0=hou_ode(y,tm,inp);
            mat y_pred=y+(dt/24.)*(55.*f0.t()-59.*f1.t()+37.*f2.t()-9.*f3.t());
            mat f_p=hou_ode(y_pred,th,inp);  // derivative at predicted state
            // Corrector uses the predicted derivative f_p alongside the 3 previous derivatives
            y_new=y+(dt/24.)*(9.*f_p.t()+19.*f0.t()-5.*f1.t()+f2.t());
            // Shift derivative history by one step
            f3=f2; f2=f1; f1=f0;
        }

        y=y_new; t+=dt;

        if(*inp.flyby_toggle){
            f0_hyp=kepler(inp.n_hyp,t,inp.e_hyp,inp.tau_hyp);
            X_s=kepler2cart(inp.a_hyp,inp.e_hyp,inp.i_hyp,inp.RAAN_hyp,inp.om_hyp,f0_hyp,inp.G,inp.Mplanet);
        }
        if(*inp.helio_toggle){
            f0_helio=kepler(inp.n_helio,t,inp.e_helio,inp.tau_helio);
            X_helio=kepler2cart(inp.a_helio,inp.e_helio,inp.i_helio,inp.RAAN_helio,inp.om_helio,f0_helio,inp.G,inp.Msolar);
        }
        if(!decimate||t>=t_next-1e-9){
            push_state(t,y,X_s,X_helio,*inp.flyby_toggle,*inp.helio_toggle,
                       inp,times_out,states_out,hyp_out,solar_out,potential_out);
            if(decimate) t_next+=out_freq;
        }
    }
}

// LGVI symplectic integrator (no perturbations, no perturber output)
static void LGVI_lib(double t0, double tf, mat x0, double h, double out_freq,
                     parameters inp,
                     std::vector<double>& times_out,
                     std::vector<double>& states_out,
                     std::vector<double>& potential_out){
    bool decimate=(out_freq>0.0);
    double t_next=t0;

    // Modified (non-standard) moments of inertia used by the LGVI Hamiltonian map.
    // IdA = 2*(tr(IA)/2 * I3 - IA), which is the "dual" MOI needed for the
    // discrete-time angular momentum update in the Lie-group variational integrator.
    (*(inp.IdA))=2.*(0.5*trace(diagmat(*(inp.IA)))*eye(3,3)-diagmat(*(inp.IA)));
    (*(inp.IdB))=2.*(0.5*trace(diagmat(*(inp.IB)))*eye(3,3)-diagmat(*(inp.IB)));

    // Initial potential partials
    mat du_dr_0(3,1),M_0(3,1);
    mat r0=x0.cols(0,2); r0.reshape(3,1);
    mat C0=x0.cols(21,29); C0.reshape(3,3); C0=C0.t();
    map_potential_partials(&C0,&r0,inp,&du_dr_0,&M_0);

    mat fab(3,1),fna(3,1),Fab(3,3),Fna(3,3),g_map(3,1),G_map(3,1),grad_G(3,3);
    fab.fill(0.4); fna.fill(0.4);

    // Lambda to record state and potential at an output point
    auto push_lgvi = [&](double t_val, const mat& yv) {
        times_out.push_back(t_val);
        for(int c=0;c<30;c++) states_out.push_back(yv(0,c));
        mat r_vec = yv.cols(0,2); r_vec.reshape(3,1);
        double R  = norm(r_vec, 2);
        mat e     = r_vec.t() / R;                       // (1,3) — matches hou_ode
        mat Cv    = yv.cols(21,29); Cv.reshape(3,3); Cv = Cv.t();
        inertia_rot(Cv, *(inp.n), inp.TB, inp.TBp);
        potential_out.push_back(
            potential(*(inp.G), *(inp.n), inp.tk, inp.a, inp.b, e, R, inp.TA, inp.TBp));
    };

    mat y=x0;
    double t=t0;

    push_lgvi(t, y);
    if(decimate) t_next+=out_freq;

    while(t<tf){
        double dt=h;
        if(t+dt>tf) dt=tf-t;
        mat y_new=zeros(1,30);
        hamiltonian_map(dt,inp,&y,&y_new,&fab,&fna,&Fab,&Fna,
                        &g_map,&G_map,&grad_G,&du_dr_0,&M_0,&x0);
        y=y_new; t+=dt;
        if(!decimate||t>=t_next-1e-9){
            push_lgvi(t, y);
            if(decimate) t_next+=out_freq;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

SimResult run_simulation(const SimConfig& cfg){
    SimResult result;

    int order  =cfg.gravity_order;
    int order_a=cfg.order_a;
    int order_b=cfg.order_b;

    double G    =cfg.G;
    double rhoAv=cfg.rhoA;
    double rhoBv=cfg.rhoB;

    // Inertia matrices
    mat IA(1,3); IA.zeros();
    mat IB(1,3); IB.zeros();
    mat IdA(3,3); IdA.zeros();
    mat IdB(3,3); IdB.zeros();
    cube TA,TB,TBp,TS,Tsun;
    field<cube> dT(3,3);

    // Expansion coefficient arrays
    mat tk(order+1,order/2+2); tk.zeros();
    sp_mat a(1,t_ind(order,order,order,order,order,order,order,order+1)+1);
    sp_mat b(1,t_ind(order,order,order,order,order,order,order,order+1)+1);

    // Perturbation workspace matrices
    mat sg_acc(3,1),acc_3BP(3,1),acc_solar(3,1);
    mat tt_1(3,1),tt_2(3,1),tt_orbit(3,1);
    sg_acc.zeros(); acc_3BP.zeros(); acc_solar.zeros();
    tt_1.zeros(); tt_2.zeros(); tt_orbit.zeros();

    // Build inertia integrals for primary
    if(cfg.a_shape==0){
        ell_mass_params_met(order,0,rhoAv,cfg.aA,cfg.bA,cfg.cA,&IA,&TA);
    } else if(cfg.a_shape==1){
        ell_mass_params_met(order,order_a,rhoAv,cfg.aA,cfg.bA,cfg.cA,&IA,&TA);
    } else {
        poly_inertia_met(order_a,rhoAv,cfg.tet_fileA,cfg.vert_fileA,&TA);
        poly_moi_met(rhoAv,cfg.tet_fileA,cfg.vert_fileA,&IA);
    }

    // Build inertia integrals for secondary
    if(cfg.b_shape==0){
        ell_mass_params_met(order,0,rhoBv,cfg.aB,cfg.bB,cfg.cB,&IB,&TB);
    } else if(cfg.b_shape==1){
        ell_mass_params_met(order,order_b,rhoBv,cfg.aB,cfg.bB,cfg.cB,&IB,&TB);
    } else {
        poly_inertia_met(order_b,rhoBv,cfg.tet_fileB,cfg.vert_fileB,&TB);
        poly_moi_met(rhoBv,cfg.tet_fileB,cfg.vert_fileB,&IB);
    }

    // Point-mass inertia integrals for perturbers
    TS.zeros(size(TA));    TS(0,0,0)=cfg.Mplanet;
    Tsun.zeros(size(TA));  Tsun(0,0,0)=cfg.Msolar;

    // Expansion coefficients
    tk_calc(order,&tk);
    a_calc(order,&a);
    b_calc(order,&b);

    // Masses and mass ratios
    double Mc=TA(0,0,0);
    double Ms=TB(0,0,0);
    double m =Mc*Ms/(Mc+Ms);
    double nu=Ms/(Ms+Mc);

    // Derived orbital parameters (local copies for pointers)
    double Gv         =G;
    double mv         =m;
    double nuv        =nu;
    int    orderv     =order;
    int    flyby_tog  =cfg.flyby_toggle;
    int    helio_tog  =cfg.helio_toggle;
    int    sg_tog     =cfg.sg_toggle;
    int    tt_tog     =cfg.tt_toggle;
    double Mplanetv   =cfg.Mplanet;
    double a_hypv     =cfg.a_hyp;
    double e_hypv     =cfg.e_hyp;
    double i_hypv     =cfg.i_hyp;
    double RAAN_hypv  =cfg.RAAN_hyp;
    double om_hypv    =cfg.om_hyp;
    double tau_hypv   =cfg.tau_hyp;
    double n_hypv     =(cfg.Mplanet>0&&cfg.a_hyp!=0)?sqrt(G*cfg.Mplanet/pow(std::abs(cfg.a_hyp),3.)):0.;
    double Msolarv    =cfg.Msolar;
    double a_heliov   =cfg.a_helio;
    double e_heliov   =cfg.e_helio;
    double i_heliov   =cfg.i_helio;
    double RAAN_heliov=cfg.RAAN_helio;
    double om_heliov  =cfg.om_helio;
    double tau_heliov =cfg.tau_helio;
    double n_heliov   =(cfg.a_helio!=0)?sqrt(G*cfg.Msolar/pow(std::abs(cfg.a_helio),3.)):0.;
    double sol_radv   =cfg.sol_rad;
    double au_defv    =cfg.au_def;
    double love1v     =cfg.love1;
    double love2v     =cfg.love2;
    double refrad1v   =cfg.refrad1;
    double refrad2v   =cfg.refrad2;
    double rhoAvp     =rhoAv;
    double rhoBvp     =rhoBv;
    double eps1v      =cfg.eps1;
    double eps2v      =cfg.eps2;
    double mean_motv  =pow(G*(cfg.Msolar+Mc+Ms)/pow(cfg.sol_rad*cfg.au_def,3.),0.5);

    // Parameters struct
    parameters inp;
    inp.G           =&Gv;
    inp.IA          =&IA;
    inp.IB          =&IB;
    inp.IdA         =&IdA;
    inp.IdB         =&IdB;
    inp.TA          =&TA;
    inp.TB          =&TB;
    inp.TBp         =&TBp;
    inp.TS          =&TS;
    inp.Tsun        =&Tsun;
    inp.a           =&a;
    inp.b           =&b;
    inp.dT          =&dT;
    inp.m           =&mv;
    inp.nu          =&nuv;
    inp.n           =&orderv;
    inp.tk          =&tk;
    inp.flyby_toggle=&flyby_tog;
    inp.helio_toggle=&helio_tog;
    inp.sg_toggle   =&sg_tog;
    inp.tt_toggle   =&tt_tog;
    inp.Mplanet     =&Mplanetv;
    inp.a_hyp       =&a_hypv;
    inp.e_hyp       =&e_hypv;
    inp.i_hyp       =&i_hypv;
    inp.RAAN_hyp    =&RAAN_hypv;
    inp.om_hyp      =&om_hypv;
    inp.tau_hyp     =&tau_hypv;
    inp.n_hyp       =&n_hypv;
    inp.Msolar      =&Msolarv;
    inp.a_helio     =&a_heliov;
    inp.e_helio     =&e_heliov;
    inp.i_helio     =&i_heliov;
    inp.RAAN_helio  =&RAAN_heliov;
    inp.om_helio    =&om_heliov;
    inp.tau_helio   =&tau_heliov;
    inp.n_helio     =&n_heliov;
    inp.sol_rad     =&sol_radv;
    inp.au_def      =&au_defv;
    inp.love1       =&love1v;
    inp.love2       =&love2v;
    inp.refrad1     =&refrad1v;
    inp.refrad2     =&refrad2v;
    inp.rhoA        =&rhoAvp;
    inp.rhoB        =&rhoBvp;
    inp.eps1        =&eps1v;
    inp.eps2        =&eps2v;
    inp.acc_3BP     =&acc_3BP;
    inp.acc_solar   =&acc_solar;
    inp.sg_acc      =&sg_acc;
    inp.tt_1        =&tt_1;
    inp.tt_2        =&tt_2;
    inp.tt_orbit    =&tt_orbit;
    inp.mean_motion =&mean_motv;

    // Initial state (1×30 row vector)
    mat x0(1,30);
    for(int i=0;i<30;i++) x0(0,i)=cfg.x0[i];

    // Run integrator
    std::vector<double> times_out,states_out,hyp_out,solar_out,potential_out;
    double out_freq=cfg.out_freq;

    switch(cfg.integ_flag){
        case 1:
            rk4_lib(cfg.t0,cfg.tf,x0,cfg.dt,out_freq,inp,
                    times_out,states_out,hyp_out,solar_out,potential_out);
            break;
        case 2:
            LGVI_lib(cfg.t0,cfg.tf,x0,cfg.dt,out_freq,inp,
                     times_out,states_out,potential_out);
            break;
        case 3:
            rk87_lib(cfg.t0,cfg.tf,x0,cfg.tol,out_freq,inp,
                     times_out,states_out,hyp_out,solar_out,potential_out);
            break;
        case 4:
            ABM_lib(cfg.t0,cfg.tf,x0,cfg.dt,out_freq,inp,
                    times_out,states_out,hyp_out,solar_out,potential_out);
            break;
        default:
            result.status="error: unknown integ_flag";
            return result;
    }

    result.times            =std::move(times_out);
    result.states           =std::move(states_out);
    result.hyp_states       =std::move(hyp_out);
    result.solar_states     =std::move(solar_out);
    result.potential_energy =std::move(potential_out);
    result.mass_primary   =Mc;
    result.mass_secondary =Ms;
    result.inertia_primary  ={IA(0,0),IA(0,1),IA(0,2)};
    result.inertia_secondary={IB(0,0),IB(0,1),IB(0,2)};

    return result;
}
