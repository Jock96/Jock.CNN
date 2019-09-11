namespace CNN.BL.Utils
{
    using CNN.BL.Constants;
    using CNN.BL.Enums;

    using System;
    using System.Text;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;

    /// <summary>
    /// Инструмент загрузки весов.
    /// </summary>
    public class WeightLoadUtil
    {
        /// <summary>
        /// Путь до файлов весов.
        /// </summary>
        private string _path;

        /// <summary>
        /// Словарь, где ключ - тип весов, значение - список весов.
        /// </summary>
        private Dictionary<WeightsType, List<string>> _weightTypeToPathDictionary;

        /// <summary>
        /// Инструмент загрузки весов.
        /// </summary>
        /// <param name="path">Путь до файлов весов.</param>
        public WeightLoadUtil(string path)
        {
            _path = path;
        }

        /// <summary>
        /// Загрузить веса.
        /// </summary>
        public void Load()
        {
            try
            {
                _weightTypeToPathDictionary = new Dictionary<WeightsType, List<string>>();
                CheckDirectory(_path);

                var outputLayerFiles = Directory.GetFiles(
                    Path.Combine(_path, FileConstants.WEIGHTS_DIRECTORY, LayersConstants.OUTPUT_LAYER_NAME)).ToList();

                CheckFiles(outputLayerFiles);
                _weightTypeToPathDictionary.Add(WeightsType.Output, outputLayerFiles);

                var hiddenLayerFiles = Directory.GetFiles(
                    Path.Combine(_path, FileConstants.WEIGHTS_DIRECTORY, LayersConstants.HIDDEN_LAYER_NAME)).ToList();

                CheckFiles(hiddenLayerFiles);
                _weightTypeToPathDictionary.Add(WeightsType.Hidden, hiddenLayerFiles);

                var coreFiles = Directory.GetFiles(
                    Path.Combine(_path, FileConstants.WEIGHTS_DIRECTORY, MatrixConstants.MATRIX_NAME)).ToList();

                CheckFiles(coreFiles);
                _weightTypeToPathDictionary.Add(WeightsType.Core, coreFiles);
            }
            catch (Exception exception)
            {
                Console.BackgroundColor = ConsoleColor.Red;
                Console.ForegroundColor = ConsoleColor.Black;

                Console.WriteLine($"{ConsoleMessageConstants.ERROR_MESSAGE} " +
                    $"{exception}!");

                Console.BackgroundColor = ConsoleColor.Black;
                Console.ForegroundColor = ConsoleColor.Green;

                Console.WriteLine(ConsoleMessageConstants.PRESS_ANY_KEY_MESSAGE);
                Console.ReadKey();

                Environment.Exit(0);
            }
        }

        /// <summary>
        /// Действия при ошибке парсинга
        /// </summary>
        private void ParseError()
        {
            Console.BackgroundColor = ConsoleColor.Red;
            Console.ForegroundColor = ConsoleColor.Black;

            Console.WriteLine($"{ConsoleMessageConstants.ERROR_MESSAGE} " +
                $"не удалось преобразовать значение веса!");

            Console.BackgroundColor = ConsoleColor.Black;
            Console.ForegroundColor = ConsoleColor.Green;

            Console.WriteLine(ConsoleMessageConstants.PRESS_ANY_KEY_MESSAGE);
            Console.ReadKey();

            Environment.Exit(0);
        }

        /// <summary>
        /// Получить данные.
        /// </summary>
        /// <returns>Вовзращает словарь, где ключ - тип весов, значение - список весов.</returns>
        public Dictionary<WeightsType, List<double>> GetData()
        {

            // TODO сделать свитч кейс и возвращать словарь словарей.
            var weightTypeToDataDictionary = new Dictionary<WeightsType, List<double>>();

            foreach (var keyValuePair in _weightTypeToPathDictionary)
            {
                var weightsValue = new List<double>();

                foreach (var path in keyValuePair.Value)
                {
                    using (var stream = File.OpenRead(path))
                    {
                        var array = new byte[stream.Length];
                        stream.Read(array, 0, array.Length);

                        var valueString = Encoding.Default.GetString(array);
                        var indexOfSeparator = 0;

                        do
                        {
                            indexOfSeparator = valueString.IndexOf(" ");

                            if (indexOfSeparator == -1)
                                continue;

                            var value = valueString.Remove(indexOfSeparator);

                            if (!double.TryParse(value, out var convertedValue))
                                ParseError();

                            weightsValue.Add(convertedValue);
                            valueString = valueString.Remove(0, value.Length + 1);
                        } while (indexOfSeparator != -1);
                    }
                }

                weightTypeToDataDictionary.Add(keyValuePair.Key, weightsValue);
            }

            return weightTypeToDataDictionary;
        }

        /// <summary>
        /// Проверить файлы.
        /// </summary>
        /// <param name="files">Полученные файлы.</param>
        private void CheckFiles(List<string> files)
        {
            if (!files.Any())
            {
                Console.BackgroundColor = ConsoleColor.Red;
                Console.ForegroundColor = ConsoleColor.Black;

                Console.WriteLine($"{ConsoleMessageConstants.ERROR_MESSAGE} " +
                    $"не удалось найти файлы весов!");

                Console.BackgroundColor = ConsoleColor.Black;
                Console.ForegroundColor = ConsoleColor.Green;

                Console.WriteLine(ConsoleMessageConstants.PRESS_ANY_KEY_MESSAGE);
                Console.ReadKey();

                Environment.Exit(0);
            }
        }

        /// <summary>
        /// Проерка директории.
        /// </summary>
        /// <param name="path">Путь.</param>
        private void CheckDirectory(string path)
        {
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
        }
    }
}
