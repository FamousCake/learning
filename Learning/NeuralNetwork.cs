using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Learning
{
    class NeuralNetwork
    {
        // Holds the number of nodes for each layer
        public List<int> L;
        
        // Holds activated values for each node in the network
        public List<Matrix> a;

        // Hold non-activated values for each node in the network
        public List<Matrix> z;

        // Hold the transition matrixes from one layer to the next
        public List<Matrix> O;

        // Delta matrixes used to compute derivatives
        public List<Matrix> d;
        public List<Matrix> D;

        // Learningn rate used by gradient descent
        public double LearningRate;

        // Basic constructor
        public NeuralNetwork(List<int> layers, double learningRate)
        {
            this.L = layers;
            this.LearningRate = learningRate;
            
            this.a = new List<Matrix>();
            this.z = new List<Matrix>();
            this.O = new List<Matrix>();
            this.D = new List<Matrix>();

            foreach (int x in layers)
            {
                a.Add(new Matrix(x, 1, 0));
                z.Add(new Matrix(x, 1, 0));
            }

            for (int i = 0; i < layers.Count - 1; i++)
            {
                Random a = new Random();
                
                // Transition matrixes are always randomized
                O.Add((new Matrix(layers[i + 1], layers[i], 1)).Apply(delegate(double x) { return a.NextDouble() * 10; }));
                D.Add(new Matrix(layers[i + 1], layers[i], 1));
            }
        }


        // Makes one "step" down during gradient descent
        public void Descend()
        {
            // The new O matrix is first computed as OO
            List<Matrix> OO = new List<Matrix>();

            // Initialization with all 0s
            for (int k = 0; k < O.Count; k++)
            {
                OO.Add(new Matrix(O[k].N, O[k].M, 0));
            }

            // Computed the new values for O
            for (int k = 0; k < O.Count; k++)
            {
                for (int i = 0; i < O[k].N; i++)
                {
                    for (int j = 0; j < O[k].M; j++)
                    {
                        OO[k][i, j] = O[k][i, j] - this.LearningRate * D[k][i,j];
                    }
                }
            }

            this.O = OO;          

        }

        
        // This here is back propagation where the derivatives are computed
        public void ComputeDerivatives(List<Matrix> X, List<Matrix> Y)
        {
            // Training examples
            int M = X.Count();

            // Default values for D
            for (int i = 0; i < this.D.Count(); i++)
            {
                D[i].Apply(delegate(double x) { return 0; });
            }

            // Computing all the derivatives
            for (int k = 0; k < M; k++)
            {
                // Step 1: Compute all errors d for every node
                List<Matrix> d = this.BackPropagate(X[k], Y[k]);

                // Step 2: Add the errors to the Derivatives
                for (int i = 0; i < this.D.Count; i++)
                {
                    this.D[i] += Matrix.Multiply(d[i + 1], this.a[i].Transpose());
                }
            }

            for (int k = 0; k < D.Count; k++)
            {
                D[k].Apply(delegate(double x) { return x / M; });
            }

        }

        // Standard forward propadation to computed output values
        public Matrix ForwardPropagate(Matrix X)
        {
            this.a[0] = new Matrix(X);

            for (int i = 1; i < a.Count; i++)
            {
                this.z[i] = Matrix.Multiply(this.O[i - 1], this.a[i - 1]);
                this.a[i] = this.g(z[i]);
            }

            return this.a.LastOrDefault();
        }

        public List<Matrix> BackPropagate(Matrix X, Matrix Y)
        {
            // 'd' Vectors are the same as 'a' vectors
            List<Matrix>d = new List<Matrix>();

            for (int i = 0; i < this.a.Count - 1; i++)
            {
                d.Add(new Matrix(this.a[i].N, this.a[i].M));
            }

            // The Final Vector in 'd' is h(x)
            d.Add(this.ForwardPropagate(X) - Y);

            // Calculate errors via the forumla d[i] = (O[i]^T) * d[i+1] .* (a[i] .* (1 - a[i]))
            for (int i = d.Count - 2; i >= 1; i--)
            {
                d[i] = Matrix.Multiply(O[i].Transpose(), d[i + 1]) * (this.a[i] * this.a[i].Apply(delegate(double x) { return 1 - x; }));
            }

            return d;
        }

        public double ComputeCostFunction(List<Matrix> X, List<Matrix> Y)
        {
            int M = X.Count();
            double S = 0;

            // Compute the error for each training example
            for (int i = 0; i < M; i++)
            {   
                S += this.computeClassificationCost(X[i], Y[i]);
            }

            S /= (-M);

            return S;
        }


        private double computeRegresionCost(Matrix X, Matrix Y)
        {
            // The formula is (h(x) - y)^2
            return (this.ForwardPropagate(X) - Y).Apply(delegate(double x) { return x * x; }).GetSum();
        }

        private double computeClassificationCost(Matrix X, Matrix Y)
        {
            // The formula is y*log(h(x) + (1-y)(log(1-h(x))
            Matrix a = Y;

            Matrix b = this.ForwardPropagate(X).Apply(delegate(double x) { return Math.Log10(x); });

            Matrix c = Y.Apply(delegate(double y) { return 1 - y; });

            Matrix d = this.ForwardPropagate(X).Apply(delegate(double x) { return Math.Log10(1 - x); });

            return (a * b + c * d).GetSum();
        }


        private Matrix g(Matrix A)
        {
            return A.Apply(delegate(double x) {
                
                if (x < -45.0) return 0.0;
                
                else if (x > 45.0) return 1.0;

                return 1.0 / (1.0 + Math.Exp(-x)); 
            });
        }
    }
}
