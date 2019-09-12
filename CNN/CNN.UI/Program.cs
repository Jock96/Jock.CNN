namespace CNN.UI
{
    using System;
    using System.IO;
    using System.Collections.Generic;
    using System.Linq;

    using BL.Helpers;

    using CNN.BL.Constants;
    using CNN.BL.Utils;
    using CNN.Core.Models;
    using CNN.Core.Layers;
    using CNN.Core;
    using CNN.Core.Utils;

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
            Console.BackgroundColor = ConsoleColor.Green;
            Console.ForegroundColor = ConsoleColor.Black;

            Console.WriteLine(ConsoleMessageConstants.HELLO_MESSAGE + "\n");

            Console.BackgroundColor = ConsoleColor.Black;
            Console.ForegroundColor = ConsoleColor.Green;

            var consoleValue = string.Empty;
            var breakFlag = false;

            do
            {
                Console.WriteLine(ConsoleMessageConstants.WORK_CHOISE_MESSAGE);

                consoleValue = Console.ReadLine();

                if (DialogConstants.LearnResults.Contains(consoleValue) ||
                    DialogConstants.RecognizeResults.Contains(consoleValue))
                    breakFlag = true;

            } while(!breakFlag);

            Console.Clear();

            if (DialogConstants.LearnResults.Contains(consoleValue))
                DoLearnWork();
            else
                DoRecognizeWork();
        }

        /// <summary>
        /// Выполнить распознавание.
        /// </summary>
        private static void DoRecognizeWork()
        {
            Console.BackgroundColor = ConsoleColor.Green;
            Console.ForegroundColor = ConsoleColor.Black;

            Console.WriteLine(ConsoleMessageConstants.RECOGNIZE_RESULT_MESSAGE);

            Console.BackgroundColor = ConsoleColor.Black;
            Console.ForegroundColor = ConsoleColor.Green;

            Console.WriteLine(ConsoleMessageConstants.LOAD_PATH_MESSAGE);

            var path = GetPathToWeightFiles();
            var weightLodUtil = new WeightLoadUtil(path);

            weightLodUtil.Load();

            var data = weightLodUtil.GetData();
            var coreToLoad = weightLodUtil.GetNewCore();

            FilterCoreModel.UpdateCore(coreToLoad);

            var breakFlag = false;
            var isFirstIteration = true;

            do
            {
                if (isFirstIteration)
                {
                    RecognizeImage(data);
                    isFirstIteration = false;
                }

                Console.WriteLine(ConsoleMessageConstants.RECOGNIZE_ANOTHER_MESSAGE);
                var dialogResult = Console.ReadLine();

                if (DialogConstants.NoResults.Contains(dialogResult))
                    breakFlag = true;

                if (DialogConstants.YesResults.Contains(dialogResult))
                    RecognizeImage(data);

            } while (!breakFlag);

            Console.WriteLine(ConsoleMessageConstants.PRESS_ANY_KEY_MESSAGE);
            Console.ReadKey();

            Console.WriteLine(ConsoleMessageConstants.EXIT_MESSAGE);
            System.Threading.Thread.Sleep(500);
            Environment.Exit(0);
        }

        /// <summary>
        /// Распознать изображение.
        /// </summary>
        /// <param name="data">Данные весов.</param>
        private static void RecognizeImage(Dictionary<BL.Enums.WeightsType, Dictionary<int, List<double>>> data)
        {
            Console.WriteLine(ConsoleMessageConstants.PATH_MESSAGE);

            var pathToImages = Console.ReadLine();

            if (string.IsNullOrEmpty(pathToImages))
                pathToImages = PathHelper.GetResourcesPath();

            if (!Directory.Exists(pathToImages))
                ErrorHelper.DirectoryError();

            var filesInDirectory = Directory.GetFiles(pathToImages).ToList();

            var imagesInDirectory = filesInDirectory.FindAll(file =>
            file.Contains(FileConstants.IMAGE_EXTENSION));

            ErrorHelper.CheckFiles(imagesInDirectory);

            var imageName = string.Empty;
            var breakFlag = false;

            do
            {
                Console.BackgroundColor = ConsoleColor.Blue;
                Console.ForegroundColor = ConsoleColor.Black;

                Console.WriteLine(ConsoleMessageConstants.IMAGES_IN_DIRECTORY_MESSAGE);

                imagesInDirectory.ForEach(imagePath =>
                Console.WriteLine(Path.GetFileNameWithoutExtension(imagePath) + " "));

                Console.Write(ConsoleMessageConstants.FILE_NAME_MESSAGE);
                imageName = Console.ReadLine();

                if (imagesInDirectory.Contains(Path.Combine(pathToImages, imageName + FileConstants.IMAGE_EXTENSION)))
                    breakFlag = true;

                Console.BackgroundColor = ConsoleColor.Black;
                Console.ForegroundColor = ConsoleColor.Green;

            } while (!breakFlag);

            var fullPath = Path.Combine(pathToImages, $"{imageName}{FileConstants.IMAGE_EXTENSION}");

            if (!data.TryGetValue(BL.Enums.WeightsType.Hidden, out var hiddenLayerWeights))
                ErrorHelper.GetDataError();

            if (!data.TryGetValue(BL.Enums.WeightsType.Output, out var outputLayerWeights))
                ErrorHelper.GetDataError();

            var imageConverterUtil = new ImageConverterUtil(new List<string> { fullPath });
            var imageMatrixList = imageConverterUtil.ConvertImagesToMatrix();

            var imageData = imageMatrixList.FirstOrDefault();

            if (imageData == null)
                ErrorHelper.GetDataError();

            var recognizeUtil = new RecognizeUtil(outputLayerWeights, hiddenLayerWeights, imageData);
            recognizeUtil.InitializeNetwork();
        }

        /// <summary>
        /// Получить путь до файлов весов.
        /// </summary>
        /// <returns>Возвращает строку пути.</returns>
        private static string GetPathToWeightFiles()
        {
            var path = Console.ReadLine();

            if (string.IsNullOrEmpty(path))
                path = PathHelper.GetResourcesPath();

            if (!Directory.Exists(path))
            {
                Console.BackgroundColor = ConsoleColor.Red;
                Console.ForegroundColor = ConsoleColor.Black;

                Console.WriteLine($"{ConsoleMessageConstants.ERROR_MESSAGE} " +
                    $"указанная директория не существует!");

                Console.BackgroundColor = ConsoleColor.Black;
                Console.ForegroundColor = ConsoleColor.Green;

                Console.WriteLine(ConsoleMessageConstants.PRESS_ANY_KEY_MESSAGE);
                Console.ReadKey();

                Environment.Exit(0);
            }

            return path;
        }

        /// <summary>
        /// Выполнить обучение.
        /// </summary>
        private static void DoLearnWork()
        {
            Console.BackgroundColor = ConsoleColor.Green;
            Console.ForegroundColor = ConsoleColor.Black;

            Console.WriteLine(ConsoleMessageConstants.LEARN_RESULT_MESSAGE);

            Console.BackgroundColor = ConsoleColor.Black;
            Console.ForegroundColor = ConsoleColor.Green;

            var pathToFiles = GetPathToFiles();
            var pathToResources = PathHelper.GetResourcesPath();

            if (string.IsNullOrEmpty(pathToFiles))
                pathToFiles = pathToResources;

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
        }

        /// <summary>
        /// Получение пути до файлов с изображениями.
        /// </summary>
        /// <returns>Возвращает путь до файлов с изображениями.</returns>
        private static string GetPathToFiles()
        {
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
    }
}
