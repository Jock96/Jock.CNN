namespace CNN.UI
{
    using System;
    using System.IO;
    using System.Collections.Generic;

    using CNN.BL.Constants;
    using CNN.BL.Utils;

    using CNN.Core.Models;
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

            var filterCore = FilterCoreModel.Initialize();
            var inputLayer = new InputLayer(matrixOfPicture);

            inputLayer.FillInputLayer();
            var inputLayerWeights = inputLayer.GetNeuronOutputs();

            var convolutionalLayer = new ConvolutionalLayer(inputLayerWeights);
            convolutionalLayer.LayerInitialize(filterCore);

            var convolutionalLayerNeurons = convolutionalLayer.GetLayerNeurons();

            // TODO: Отладка, убрать.
            GetDebugInfo(filterCore, inputLayerWeights, convolutionalLayerNeurons);

            Console.ReadKey();
        }

        /// <summary>
        /// Выводит отладочную информацию.
        /// </summary>
        /// <param name="filterCore">Ядро фильра.</param>
        /// <param name="inputLayerWeights">Вывод входного слоя.</param>
        /// <param name="convolutionalLayerNeurons">Нейроны свёрточного слоя.</param>
        private static void GetDebugInfo(double[,] filterCore, 
            Dictionary<string, double> inputLayerWeights, 
            List<NeuronModel> convolutionalLayerNeurons)
        {
            Console.WriteLine("Ядро фильтра:");

            foreach (var value in filterCore)
                Console.Write(value.ToString() + " ");

            Console.WriteLine("\nЗначения входного слоя:");

            foreach (var value in inputLayerWeights)
                Console.WriteLine(value.Key.ToString() + ": " + value.Value.ToString());

            foreach (var value in convolutionalLayerNeurons)
                Console.WriteLine(convolutionalLayerNeurons.IndexOf(value).ToString() +
                    ": " + value.Output.ToString());
        }
    }
}
