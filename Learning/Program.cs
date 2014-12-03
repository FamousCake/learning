using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Learning
{
    class Program
    {
        static List<Matrix> ReadData (string filename)
        {
            List<Matrix> L = new List<Matrix>();

            string line;
            System.IO.StreamReader file = new System.IO.StreamReader(filename);
            while ((line = file.ReadLine()) != null)
            {
                string[] a = line.Split(' ');
                double[,] b = new double[a.Length,1];

                for (int i = 0; i < a.Length; i++)
                {
                    b[i, 0] = Convert.ToDouble(a[i]);
                }

                L.Add(new Matrix(b, a.Length, 1));
            }

            file.Close();

            return L;
        }

        static void Main(string[] args)
        {
            List<Matrix> X = ReadData("input.txt");
            List<Matrix> Y = ReadData("output.txt");

            List<int> layers = new List<int>();


            layers.Add(2);
            layers.Add(10);
            layers.Add(4);
            layers.Add(9);
            layers.Add(7);
            layers.Add(1);

            NeuralNetwork N = new NeuralNetwork(layers,0.00005);
         

            N.ForwardPropagate(X[1]);

            Console.WriteLine(N.ComputeCostFunction(X, Y));

          
            int tt = 20000;


            for (int i = 0; i < tt; i++)
            {
                N.ComputeDerivatives(X, Y);
                N.Descend();
                
            }

            Console.WriteLine("Const : " + N.ComputeCostFunction(X, Y));

            


            Console.Write("\n\nPress any key to continue...");
            Console.ReadLine();
        }
    }
}

