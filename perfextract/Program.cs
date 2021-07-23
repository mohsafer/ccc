﻿using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using Aspose.Cells;
namespace Test
{
    static class Program
    {
        //private static string _processName;
        private static Dictionary<string, float[]> data = new Dictionary<string, float[]>();
        static void Main(string[] args)
        {
            Workbook wb = new Workbook();
            Worksheet sheet = wb.Worksheets[0];
            Cell cell;
            var threadFinishEvents = new List<EventWaitHandle>();
            var manualResetEventSlim = new ManualResetEventSlim();
            var myProcess = Process.Start(args[0]);
            if (myProcess == null) return;
            var processName = GetInstanceNameForProcessId(myProcess.Id);
            if (string.IsNullOrEmpty(processName)) return;
            string[] countersN =
            {
                "% Privileged Time", "Handle Count", "IO Read Operations/sec", "IO Data Operations/sec",
                "IO Write Operations/sec", "IO Other Operations/sec", "IO Read Bytes/sec", "IO Write Bytes/sec", 
                "IO Data Bytes/sec", "IO Other Bytes/sec", "Page Faults/sec", "Page File Bytes Peak", "Page File Bytes",
                "Pool Paged Bytes", "Pool Nonpaged Bytes", "Private Bytes", "Priority Base", "Thread Count", "Virtual Bytes Peak",
                "Virtual Bytes", "Working Set Peak", "Working Set", "Working Set - Private"};
            Console.WriteLine(countersN.Length);
            var counters = new ArrayList();
            foreach (var counter in countersN)
            {
                counters.Add(new PerformanceCounter("Process", counter, processName));
                data.Add(counter, new float[60]);
            }

            foreach (PerformanceCounter counter in counters)
            {
                var threadFinish = new EventWaitHandle(false, EventResetMode.ManualReset);
                threadFinishEvents.Add(threadFinish);
                
                var thread = new Thread((cnt) =>
                {
                    (cnt as PerformanceCounter).NextValue();
                    manualResetEventSlim.Wait();
                    GetCounterData(cnt as PerformanceCounter);
                    threadFinish.Set();
                });
                thread.Start(counter);
            }
            manualResetEventSlim.Set();
            Mutex.WaitAll(threadFinishEvents.ToArray(), 60000);
            int c = 0;
            foreach (KeyValuePair<string, float[]> kvp in data)
            {
                cell = sheet.Cells[((char) ('A' + c)).ToString() + 1];
                cell.PutValue(kvp.Key);
                for (int i = 2; i <= 61; i++)
                {
                    cell = sheet.Cells[((char) ('A' + c)).ToString() + i];
                    cell.PutValue(kvp.Value[i - 2]);
                }

                c++;
            }
            wb.Save("dataset.xlsx", SaveFormat.Xlsx);
            Console.WriteLine("All done");

        }

        private static void GetCounterData(PerformanceCounter counter)
        {
            for (int i = 0; i < 60; i++)
            {
                Thread.Sleep(500);

                float val = counter.NextValue();

                //Console.WriteLine(counter.CounterName + ": " + val);
                data[counter.CounterName][i] = val;
            }
        }
        

        private static string GetInstanceNameForProcessId(int processId)
        {
            var process = Process.GetProcessById(processId);
            string processName = Path.GetFileNameWithoutExtension(process.ProcessName);

            PerformanceCounterCategory cat = new PerformanceCounterCategory("Process");
            string[] instances = cat.GetInstanceNames()
                .Where(inst => inst.StartsWith(processName))
                .ToArray();

            foreach (string instance in instances)
            {
                using PerformanceCounter cnt = new PerformanceCounter("Process",
                    "ID Process", instance, true);
                int val = (int)cnt.RawValue;
                if (val == processId)
                {
                    return instance;
                }
            }
            return null;
        }
    }
    
}