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


            layers.Add(1);
            layers.Add(3);
            layers.Add(1);

            NeuralNetwork N = new NeuralNetwork(layers,0.0005);
          
            int tt = 20000;

            double last;

            foreach (Matrix x in N.O)
            {
                x.Print();
                Console.WriteLine();
            }

            do
            {
                last = N.ComputeCostFunction(X, Y);
                N.ComputeDerivatives(X, Y);
                N.Descend();
                Console.WriteLine("Const : " + N.ComputeCostFunction(X, Y));
                tt--;
            } while (last > N.ComputeCostFunction(X, Y));

            Console.WriteLine("Const : " + N.ComputeCostFunction(X, Y));

            foreach (Matrix x in N.O)
            {
                x.Print();
                Console.WriteLine();
            }

            // N.ForwardPropagate(new Matrix(1, 1, -100)).Print();
            


            Console.Write("\n\nPress any key to continue...");
            Console.ReadLine();
        }
    }
}

