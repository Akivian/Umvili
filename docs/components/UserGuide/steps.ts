import type { StepItem } from './StepNav';

export const GUIDE_STEPS: StepItem[] = [
  {
    id: 1,
    label: 'Clone & Install',
    command: 'git clone https://github.com/Akivian/Umvili.git && cd Umvili && pip install -r requirements.txt',
    output: [
      "Cloning into 'Umvili'... done.",
      'Installing dependencies... [SUCCESS]',
    ],
  },
  {
    id: 2,
    label: 'Execution',
    command: 'python main.py',
    output: [
      'Starting MARL Simulation Engine...',
      'UI initialized at localhost:8080',
    ],
  },
  {
    id: 3,
    label: 'Advanced Config',
    command: 'python main.py --config config/default.json',
    output: [
      'Applying override: default.json',
      'Overriding grid_size to 100...',
    ],
  },
];
