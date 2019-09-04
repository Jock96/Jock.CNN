namespace CNN.Core.Utils
{
    using System;
    using System.Collections.Generic;

    using BL.Enums;

    using CNN.Core.Layers;

    /// <summary>
    /// Инструмент обучения нейронной сети.
    /// </summary>
    public class LearningUtil
    {
        /// <summary>
        /// Список слоёв нейронной сети.
        /// </summary>
        private List<Layer> _layers;

        /// <summary>
        /// Инструмент обучения нейронной сети.
        /// </summary>
        /// <param name="layers">Список слоёв нейронной сети.</param>
        /// <param name="configuration">Конфигурация нейронной сети.</param>
        public LearningUtil(List<Layer> layers, Configuration configuration)
        {
            _layers = layers;
        }

        // Метод обучения (эпохи и итерации)
        // Метод обратного распространения.
        // Для выходного.
        // Для скрытого.

        // Для свёртончного - 
        // подсчитывается общая ошибка для каждого значения
        // в ядре фильтра (в общем их 9 у меня), 
        // К каждому из ядра прибавляется общая сумма
        // соответствующей ячейки в своей позиции

        //Просчёт ошибки
    }
}
