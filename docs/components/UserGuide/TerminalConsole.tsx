'use client';

import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Copy } from 'lucide-react';
import { cn } from '../../common/utils';
import type { StepItem } from './StepNav';

const TYPEWRITER_MS = 40;
const EXEC_DELAY_MS = 300;

interface TerminalConsoleProps {
  step: StepItem;
  /** Copy callback receives the command */
  onCopy?: (command: string) => void;
}

export function TerminalConsole({ step, onCopy }: TerminalConsoleProps) {
  const [displayedCommand, setDisplayedCommand] = useState('');
  const [showOutput, setShowOutput] = useState(false);
  const [outputIndex, setOutputIndex] = useState(0);
  const [isTyping, setIsTyping] = useState(true);

  const copyCommand = useCallback(() => {
    navigator.clipboard?.writeText(step.command);
    onCopy?.(step.command);
  }, [step.command, onCopy]);

  useEffect(() => {
    setDisplayedCommand('');
    setShowOutput(false);
    setOutputIndex(0);
    setIsTyping(true);
  }, [step.id]);

  useEffect(() => {
    if (!isTyping) return;
    if (displayedCommand.length >= step.command.length) {
      setIsTyping(false);
      const t = setTimeout(() => setShowOutput(true), EXEC_DELAY_MS);
      return () => clearTimeout(t);
    }
    const t = setTimeout(() => {
      setDisplayedCommand(step.command.slice(0, displayedCommand.length + 1));
    }, TYPEWRITER_MS);
    return () => clearTimeout(t);
  }, [displayedCommand, step.command, isTyping]);

  useEffect(() => {
    if (!showOutput || outputIndex >= step.output.length) return;
    const t = setTimeout(() => setOutputIndex((i) => i + 1), 400);
    return () => clearTimeout(t);
  }, [showOutput, outputIndex, step.output.length]);

  return (
    <div
      className={cn(
        'rounded-sm overflow-hidden',
        'bg-zinc-950/30 backdrop-blur-md',
        'border border-zinc-800',
        'shadow-[inset_0_1px_1px_0_rgba(255,255,255,0.05)]'
      )}
    >
      {/* Header: traffic lights + copy */}
      <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-2.5">
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-[#ff5f56]" />
          <span className="w-2.5 h-2.5 rounded-full bg-[#ffbd2e]" />
          <span className="w-2.5 h-2.5 rounded-full bg-[#27c93f]" />
        </div>
        <button
          type="button"
          onClick={copyCommand}
          className="p-1.5 rounded-sm text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/50 transition-colors"
          aria-label="Copy command"
        >
          <Copy className="w-3.5 h-3.5" strokeWidth={1.5} />
        </button>
      </div>

      {/* Body: dot grid bg + content */}
      <div
        className="relative min-h-[180px] p-4 font-mono text-sm overflow-x-auto"
        style={{
          backgroundImage: 'radial-gradient(circle, rgba(255,255,255,0.04) 0.5px, transparent 0.5px)',
          backgroundSize: '24px 24px',
          backgroundColor: 'rgba(0,0,0,0.8)',
        }}
      >
        <AnimatePresence mode="wait">
          <motion.div
            key={step.id}
            initial={{ opacity: 0, filter: 'blur(4px)' }}
            animate={{ opacity: 1, filter: 'blur(0px)' }}
            exit={{ opacity: 0, filter: 'blur(4px)' }}
            transition={{ duration: 0.2 }}
            className="space-y-2"
          >
            {/* Command line */}
            <div className="flex items-baseline gap-2">
              <span className="text-zinc-500 shrink-0">$</span>
              <span className="text-zinc-200 break-all">
                {displayedCommand}
                {isTyping && (
                  <span
                    className="inline-block w-3 h-px ml-0.5 bg-zinc-400 align-middle animate-[blink_1s_ease-in-out_infinite]"
                    aria-hidden
                  />
                )}
              </span>
            </div>

            {/* Output lines */}
            {showOutput &&
              step.output.slice(0, outputIndex).map((line, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -4 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="text-zinc-500"
                >
                  &gt; {line}
                </motion.div>
              ))}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
