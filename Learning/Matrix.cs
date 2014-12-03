using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Learning
{
    class Matrix
    {
        public double[,] data;
        public int N, M;

        public delegate double applyCallback(double x);
        public delegate double applyCallbackElementWise(double x, double y);

        public double this[int i, int j]
        {
            get
            {
                return data[i, j];
            }

            set { data[i, j] = value; }
        }

        public Matrix(int N, int M, int initialValue = 1)
        {
            this.M = M;
            this.N = N;

            data = new double[N, M];

            for (int i = 0; i < this.N; i++)
            {
                for (int j = 0; j < this.M; j++)
                {
                    data[i, j] = initialValue;
                }
            }
        }

        public Matrix(Matrix A)
        {
            this.M = A.M;
            this.N = A.N;

            this.data = new double[N, M];

            for (int i = 0; i < this.N; i++)
            {
                for (int j = 0; j < this.M; j++)
                {
                    this.data[i, j] = A[i, j];
                }
            }
        }

        public Matrix(double[,] A, int N, int M)
        {
            this.N = N;
            this.M = M;            

            this.data = new double[N, M];

            for (int i = 0; i < this.N; i++)
            {
                for (int j = 0; j < this.M; j++)
                {
                    this.data[i, j] = A[i, j];
                }
            }
        }

        public static Matrix Multiply(Matrix A, Matrix B)
        {
            if (A.M != B.N)
            {
                throw new Exception("Matrix Dimentions not Equal!");
            }

            Matrix C = new Matrix(A.N, B.M);

            for (int i = 0; i < C.N; i++)
            {
                for (int j = 0; j < C.M; j++)
                {
                    C[i, j] = 0;

                    for (int k = 0; k < A.M; k++)
                    {
                        C.data[i, j] += A.data[i, k] * B.data[k, j];
                    }
                }
            }

            return C;
        }

        public static Matrix Add(Matrix A, Matrix B)
        {
            Matrix C = new Matrix(A.N, A.M);

            for (int i = 0; i < C.N; i++)
            {
                for (int j = 0; j < C.M; j++)
                {
                    C[i, j] = A[i, j] + B[i, j];
                }
            }

            return C;
        }

        public static Matrix Multiply(Matrix A, int x)
        {
            Matrix C = new Matrix(A.N, A.M);

            for (int i = 0; i < C.N; i++)
            {
                for (int j = 0; j < C.M; j++)
                {
                    C[i, j] = A[i, j] * x;
                }
            }

            return C;
        }


        public Matrix Transpose()
        {
            Matrix B = new Matrix(this.M, this.N);

            for (int i = 0; i < this.N; i++)
            {
                for (int j = 0; j < this.M; j++)
                {
                    B[j, i] = this.data[i, j];
                }
            }

            return B;
        }

        public Matrix Apply(applyCallback callback)
        {
            Matrix A = new Matrix(this.N, this.M);

            for (int i = 0; i < A.N; i++)
            {
                for (int j = 0; j < A.M; j++)
                {
                    A.data[i, j] = callback(this.data[i, j]);
                }
            }

            return A;
        }

        public Matrix ApplyElementWise(Matrix A, applyCallbackElementWise callback)
        {
            Matrix B = new Matrix(this.N, this.M);

            for (int i = 0; i < this.N; i++)
            {
                for (int j = 0; j < this.M; j++)
                {
                    B.data[i, j] = callback(this.data[i, j], A[i, j]);
                }
            }

            return B;

        }

        public void Print()
        {
            for (int i = 0; i < this.N; i++)
            {
                for (int j = 0; j < this.M; j++)
                {
                    Console.Write(this.data[i, j] + " ");
                }
                Console.WriteLine();
            }
        }

        public Matrix SubMatrix(int N1, int N2, int M1, int M2)
        {
            Matrix A = new Matrix(N2 - N1 + 1, M2 - M1 + 1);

            for (int i = N1; i <= N2; i++)
            {
                for (int j = M1; j <= M2; j++)
                {
                    A[i - N1, j - M1] = this.data[i, j];
                }
            }

            return A;

        }

        public double GetSum()
        {
            double S = 0;

            for (int i = 0; i < this.N; i++)
            {
                for (int j = 0; j < this.M; j++) {
                    S += this.data[i,j];
                }
            }

            return S;
        }



        public static Matrix operator +(Matrix A, Matrix B)
        {
            Matrix C = new Matrix(A.N, A.M);
            for (int i = 0; i < A.N; i++)
            {
                for (int j = 0; j < A.M; j++)
                {
                    C[i, j] = A[i, j] + B[i, j];
                }
            }

            return C;
        }

        public static Matrix operator -(Matrix A, Matrix B)
        {
            Matrix C = new Matrix(A.N, A.M);
            for (int i = 0; i < A.N; i++)
            {
                for (int j = 0; j < A.M; j++)
                {
                    C[i, j] = A[i, j] - B[i, j];
                }
            }

            return C;
        }

        public static Matrix operator *(Matrix A, Matrix B)
        {
            Matrix C = new Matrix(A.N, A.M);
            for (int i = 0; i < A.N; i++)
            {
                for (int j = 0; j < A.M; j++)
                {
                    C[i, j] = A[i, j] * B[i, j];
                }
            }

            return C;
        }

        public static Matrix operator /(Matrix A, Matrix B)
        {
            Matrix C = new Matrix(A.N, A.M);
            for (int i = 0; i < A.N; i++)
            {
                for (int j = 0; j < A.M; j++)
                {
                    C[i, j] = A[i, j] / B[i, j];
                }
            }

            return C;
        }
    }
}
