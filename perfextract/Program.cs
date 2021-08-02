﻿using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using Aspose.Cells;
using System.Management;

namespace perfextract
{
    static class Program
    {
        private static List<Dictionary<string, float[]>> Data = new List<Dictionary<string, float[]>>();
        private static int Count = 0;
        private static List<EventWaitHandle> threadFinishEvents = new List<EventWaitHandle>();
        private static List<int> IdList = new List<int>();

        private static readonly string[] CountersList =
        {
            "% Privileged Time", "Handle Count", "IO Read Operations/sec", "IO Data Operations/sec",
            "IO Write Operations/sec", "IO Other Operations/sec", "IO Read Bytes/sec", "IO Write Bytes/sec",
            "IO Data Bytes/sec", "IO Other Bytes/sec", "Page Faults/sec", "Page File Bytes Peak",
            "Page File Bytes",
            "Pool Paged Bytes", "Pool Nonpaged Bytes", "Private Bytes", "Priority Base", "Thread Count",
            "Virtual Bytes Peak",
            "Virtual Bytes", "Working Set Peak", "Working Set", "Working Set - Private"
        };

        static void Main(string[] args)
        {
            var myProcess = Process.Start(args[0]);
            Console.WriteLine("Started: " + DateTime.Now.ToString());
            if (myProcess == null) return;
            StartThread(myProcess.Id);
            Mutex.WaitAll(threadFinishEvents.ToArray(), 600000);
        }

        private static void StartThreadForChild(int id)
        {
            for (int i = 0; i < 10; i++)
            {
                var child = GetChildProcesses(id);
                if (child.Count > 0)
                {
                    foreach (var process in child)
                    {
                        var _id = Convert.ToInt32(process.GetPropertyValue("ProcessId"));
                        if (!IdList.Contains(_id))
                        {
                            StartThread(_id);
                        }
                    }
                }

                Thread.Sleep(3000);
            }
        }

        private static void StartThread(int proc)
        {
            var threadFinish = new EventWaitHandle(false, EventResetMode.ManualReset);
            threadFinishEvents.Add(threadFinish);
            var thread = new Thread(process =>
            {
                Extract(process as Process);
                threadFinish.Set();
            });
            thread.Start(Process.GetProcessById(proc));
        }

        private static void Extract(Process myProcess)
        {
            Workbook wb = new Workbook();
            Worksheet sheet = wb.Worksheets[0];
            var manualResetEventSlim = new ManualResetEventSlim();
            List<EventWaitHandle> threadFinishEventsInternal = new List<EventWaitHandle>();
            if (myProcess == null) return;
            var processName = GetInstanceNameForProcessId(myProcess.Id);
            Thread.Sleep(1000);
            var child = GetChildProcesses(myProcess.Id);
            if (child.Count > 0)
            {
                foreach (var process in child)
                {
                    StartThread(Convert.ToInt32(process.GetPropertyValue("ProcessId")));
                }
            }

            if (string.IsNullOrEmpty(processName)) return;
            IdList.Add(myProcess.Id);
            Console.WriteLine(processName);
            var counters = new ArrayList();
            var count = Count++;
            Data.Add(new Dictionary<string, float[]>());
            foreach (var counter in CountersList)
            {
                counters.Add(new PerformanceCounter("Process", counter, processName));
                Data[count].Add(counter, new float[60]);
            }

            Console.WriteLine("Reading: " + count + " " + DateTime.Now.ToString());
            foreach (PerformanceCounter counter in counters)
            {
                var threadFinish = new EventWaitHandle(false, EventResetMode.ManualReset);
                threadFinishEventsInternal.Add(threadFinish);

                var thread = new Thread((cnt) =>
                {
                    (cnt as PerformanceCounter).NextValue();
                    manualResetEventSlim.Wait();
                    GetCounterData(cnt as PerformanceCounter, count);
                    threadFinish.Set();
                });
                thread.Start(counter);
            }

            var _threadFinish = new EventWaitHandle(false, EventResetMode.ManualReset);
            threadFinishEventsInternal.Add(_threadFinish);

            var _thread = new Thread((id) =>
            {
                manualResetEventSlim.Wait();
                StartThreadForChild((int) id);
                _threadFinish.Set();
            });
            _thread.Start(myProcess.Id);
            manualResetEventSlim.Set();
            Mutex.WaitAll(threadFinishEventsInternal.ToArray(), 60000);
            IdList.Remove(myProcess.Id);
            Console.WriteLine("Writing: " + count + " " + DateTime.Now.ToString());
            var c = 0;
            foreach (KeyValuePair<string, float[]> kvp in Data[count])
            {
                var cell = sheet.Cells[((char) ('A' + c)).ToString() + 1];
                cell.PutValue(kvp.Key);
                for (int i = 2; i <= 61; i++)
                {
                    cell = sheet.Cells[((char) ('A' + c)).ToString() + i];
                    cell.PutValue(kvp.Value[i - 2]);
                }

                c++;
            }

            wb.Save("dataset" + count + ".csv", SaveFormat.Csv);
            Console.WriteLine("All done");
        }


        private static void GetCounterData(PerformanceCounter counter, int index)
        {
            for (int i = 0; i < 60; i++)
            {
                Thread.Sleep(500);
                float val = counter.NextValue();
                Data[index][counter.CounterName][i] = val;
            }
        }

        static ManagementObjectCollection GetChildProcesses(int parentId)
        {
            var query = "SELECT * FROM Win32_Process WHERE ParentProcessId = " + parentId;
            var searcher = new ManagementObjectSearcher(query);
            var processList = searcher.Get();
            return processList;
        }

        private static string GetInstanceNameForProcessId(int processId)
        {
            try
            {
                var process_ = Process.GetProcessById(processId);
                Path.GetFileNameWithoutExtension(process_.ProcessName);
            }
            catch (Exception e)
            {
                return null;
            }

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
                int val = (int) cnt.RawValue;
                if (val == processId)
                {
                    return instance;
                }
            }

            return null;
        }
    }
}