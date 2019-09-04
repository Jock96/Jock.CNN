namespace CNN.UI
{
    using System;
    using System.IO;

    using CNN.BL.Constants;
    using CNN.BL.Utils;

    using CNN.Core;
    using CNN.Core.Layers;

    /// <summary>
    /// Класс входной точки приложения.
    /// </summary>
    class Program
    {
        /// <summary>
        /// Точка входа приложения.
        /// </summary>
        static void Main(string[] args)
        {
            Console.ForegroundColor = ConsoleColor.Green;

            Console.WriteLine(ConsoleMessageConstants.HELLO_MESSAGE);
            Console.WriteLine(ConsoleMessageConstants.PRESS_ANY_KEY_MESSAGE);

            Console.ReadKey();

            Console.WriteLine(ConsoleMessageConstants.FILE_NAME_MESSAGE);

            var fileName = Console.ReadLine();
            var pathToResources = BL.Helpers.PathHelper.GetResourcesPath();

            var path = Path.Combine(pathToResources, fileName + FileConstants.IMAGE_EXTENSION);

            var converter = new ImageConverterUtil(path);
            var matrixOfPicture = converter.ConvertImageToMatrix();

            var filterCore = FilterCore.Initialize();
            var inputLayer = new InputLayer(matrixOfPicture);

            inputLayer.FillInputLayer();
            var inputLayerWeights = inputLayer.GetNeuronOutputs();

            // TODO: Отладка, убрать.
            GetDebugInfo(filterCore, inputLayerWeights);

            Console.ReadKey();
        }

        /// <summary>
        /// Выводит отладочную информацию.
        /// </summary>
        /// <param name="filterCore">Ядро фильра.</param>
        /// <param name="inputLayerWeights">Вывод входного слоя.</param>
        private static void GetDebugInfo(double[,] filterCore, System.Collections.Generic.Dictionary<string, double> inputLayerWeights)
        {
            Console.WriteLine("Ядро фильтра:");

            foreach (var value in filterCore)
                Console.Write(value.ToString() + " ");

            Console.WriteLine("\nЗначения входного слоя:");

            foreach (var value in inputLayerWeights)
            {
                Console.WriteLine(value.Key.ToString() + ": " + value.Value.ToString());
            }
        }
    }
}
