namespace CNN.UI
{
    using System;
    using System.IO;
    using System.Collections.Generic;

    using CNN.BL.Constants;
    using CNN.BL.Utils;

    using CNN.Core.Models;
    using CNN.Core.Layers;
    using CNN.Core;
    using CNN.Core.Utils;
    using System.Linq;

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
            var pathToFiles = GetPathToFiles();
            var pathToResources = BL.Helpers.PathHelper.GetResourcesPath();

            if (string.IsNullOrEmpty(pathToFiles))
                pathToFiles = pathToResources;

            //var path = Path.Combine(pathToResources, fileName + FileConstants.IMAGE_EXTENSION);

            var images = Directory.GetFiles(pathToFiles).ToList();

            var converter = new ImageConverterUtil(images);
            var listOfPicturesMatrix = converter.ConvertImagesToMatrix();

            var iterationsCount = listOfPicturesMatrix.Count;

            var configuration = new Configuration
            {
                Alpha = 3.1,
                Epsilon = 3.1,
                EpochCount = 3,
                IdealResult = 1,
                IterationsInEpochCount = iterationsCount
            };

            var layers = new List<Layer>();
            var filterCore = FilterCoreModel.Initialize();

            LayersInitialize(listOfPicturesMatrix, layers, filterCore,
                out Dictionary<string, double> inputLayerWeights,
                out List<NeuronModel> convolutionalLayerNeurons,
                out List<NeuronModel> hiddenLayerNeurons,
                out NeuronModel outputNeuron);

            var learningUtil = new LearningUtil(layers, configuration, listOfPicturesMatrix);
            learningUtil.StartToLearn();

            return;

            // TODO: Отладка, убрать.
            GetDebugInfo(filterCore, inputLayerWeights,
                convolutionalLayerNeurons, hiddenLayerNeurons, outputNeuron);

            Console.ReadKey();
        }

        /// <summary>
        /// Получение пути до файлов с изображениями.
        /// </summary>
        /// <returns>Возвращает путь до файлов с изображениями.</returns>
        private static string GetPathToFiles()
        {
            Console.ForegroundColor = ConsoleColor.Green;

            Console.WriteLine(ConsoleMessageConstants.HELLO_MESSAGE);
            Console.WriteLine(ConsoleMessageConstants.PRESS_ANY_KEY_MESSAGE);

            Console.ReadKey();

            Console.ForegroundColor = ConsoleColor.Cyan;

            Console.Write($"\n{ConsoleMessageConstants.PATH_MESSAGE}");

            var fileName = Console.ReadLine();
            Console.WriteLine();

            Console.ForegroundColor = ConsoleColor.Green;
            return fileName;
        }

        /// <summary>
        /// Инициализация слоёв нейронной сети.
        /// </summary>
        /// <param name="listOfPicturesmatrix">Список матриц изображений.</param>
        /// <param name="layers">Список слоёв.</param>
        /// <param name="filterCore">Ядро фильтра.</param>
        /// <param name="inputLayerNeurons">Нейроны выходного слоя.</param>
        /// <param name="convolutionalLayerNeurons">Нейроны свёрточного слоя.</param>
        /// <param name="hiddenLayerNeurons">Нейроны скрытого слоя.</param>
        /// <param name="outputNeuron">Нейроны выходного слоя.</param>
        private static void LayersInitialize(List<double[,]> listOfPicturesmatrix, List<Layer> layers, 
            double[,] filterCore, out Dictionary<string, double> inputLayerNeurons,
            out List<NeuronModel> convolutionalLayerNeurons, out List<NeuronModel> hiddenLayerNeurons,
            out NeuronModel outputNeuron)
        {
            var firstDataSet = listOfPicturesmatrix.First();
            var inputLayer = new InputLayer(firstDataSet);

            inputLayer.Initialize();
            layers.Add(inputLayer);

            inputLayerNeurons = inputLayer.GetLayerNeurons();
            var convolutionalLayer = new ConvolutionalLayer(inputLayerNeurons);

            convolutionalLayer.Initialize(filterCore);
            layers.Add(convolutionalLayer);

            convolutionalLayerNeurons = convolutionalLayer.GetLayerNeurons();
            var hiddenLayer = new HiddenLayer(convolutionalLayerNeurons);

            hiddenLayer.Initialize();
            layers.Add(hiddenLayer);

            hiddenLayerNeurons = hiddenLayer.GetLayerNeurons();
            var outputLayer = new OutputLayer(hiddenLayerNeurons);

            outputLayer.Initilize();
            layers.Add(outputLayer);

            outputNeuron = outputLayer.GetOutputNeuron();
        }

        /// <summary>
        /// Выводит отладочную информацию.
        /// </summary>
        /// <param name="filterCore">Ядро фильра.</param>
        /// <param name="inputLayerWeights">Вывод входного слоя.</param>
        /// <param name="convolutionalLayerNeurons">Нейроны свёрточного слоя.</param>
        /// <param name="hiddenLayerNeurons">Нейроны скрытого слоя.</param>
        /// <param name="outputNeuron">Выходной нейрон.</param>
        private static void GetDebugInfo(double[,] filterCore, 
            Dictionary<string, double> inputLayerWeights, 
            List<NeuronModel> convolutionalLayerNeurons,
            List<NeuronModel> hiddenLayerNeurons, NeuronModel outputNeuron)
        {
            Console.WriteLine("Ядро фильтра:");

            foreach (var value in filterCore)
                Console.Write(value.ToString() + " ");

            Console.WriteLine("\nЗначения входного слоя:");

            foreach (var value in inputLayerWeights)
                Console.WriteLine(value.Key.ToString() + ": " + value.Value.ToString());

            Console.WriteLine("\nЗначения свёрточного слоя:");

            foreach (var value in convolutionalLayerNeurons)
                Console.WriteLine(convolutionalLayerNeurons.IndexOf(value).ToString() +
                    ": " + value.Output.ToString());

            Console.WriteLine("\nЗначения скрытого слоя:");

            foreach (var value in hiddenLayerNeurons)
                Console.WriteLine(hiddenLayerNeurons.IndexOf(value).ToString() +
                    ": " + value.Output.ToString());

            Console.WriteLine("\nЗначение выходного нейрона:");
            Console.WriteLine(outputNeuron.Output);
        }
    }
}
