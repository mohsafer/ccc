﻿using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using Aspose.Cells;
using System.Management;
using CommandLine;

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

        private static string _output;
        public class Options
        {
            [Option('p', "PID", Required = false)] 
            public int Pid { get; set; }

            [Option('f', "filepath", Required = false, Default = "")]
            public string Filepath { get; set; }

            [Option('c', "child", Required = false, Default = false)]
            public bool child { get; set; }
            
            [Option('o', "Output path", Required = false, Default = "")]
            public string Output { get; set; }
        }

        static void Main(string[] args)
        {
            Parser.Default.ParseArguments<Options>(args)
                .WithParsed(o =>
                {
                    _output = o.Output;
                    if (!string.IsNullOrEmpty(o.Filepath))
                    {
                        var myProcess = Process.Start(o.Filepath);
                        Console.WriteLine("Started: " + DateTime.Now);
                        if (myProcess == null) return;
                        StartThread(myProcess.Id, o.child);
                    }
                    else
                    {
                        Console.WriteLine("Started: " + DateTime.Now);
                        StartThread(o.Pid, o.child);
                    }
                });
            Mutex.WaitAll(threadFinishEvents.ToArray(), 600000);
        }

        private static  void StartThreadForChild(int id)
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

        private static void StartThread(int proc, bool checkChild = true)
        {
            var threadFinish = new EventWaitHandle(false, EventResetMode.ManualReset);
            threadFinishEvents.Add(threadFinish);
            var thread = new Thread(process =>
            {
                Extract(process as Process, checkChild);
                threadFinish.Set();
            });
            thread.Start(Process.GetProcessById(proc));
        }

        private static  void Extract(Process myProcess, bool checkChild = true)
        {
            Workbook wb = new Workbook();
            Worksheet sheet = wb.Worksheets[0];
            IdList.Add(myProcess.Id);
            bool done = false;
            var manualResetEventSlim = new ManualResetEventSlim();
            List<EventWaitHandle> threadFinishEventsInternal = new List<EventWaitHandle>();
            if (myProcess == null) return;
            var processName = GetInstanceNameForProcessId(myProcess.Id);
            if (checkChild)
            {
                Thread.Sleep(1000);
                var child = GetChildProcesses(myProcess.Id);
                if (child.Count > 0)
                {
                    foreach (var process in child)
                    {
                        StartThread(Convert.ToInt32(process.GetPropertyValue("ProcessId")));
                    }
                }
            }

            if (string.IsNullOrEmpty(processName)) return;
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
                    manualResetEventSlim.Wait();
                    done = GetCounterData(cnt as PerformanceCounter, count);
                    threadFinish.Set();
                });
                thread.Start(counter);
            }

            if (checkChild)
            {
                var _threadFinish = new EventWaitHandle(false, EventResetMode.ManualReset);
                threadFinishEventsInternal.Add(_threadFinish);

                var _thread = new Thread(id =>
                {
                    manualResetEventSlim.Wait();
                    StartThreadForChild((int) id);
                    _threadFinish.Set();
                });
                _thread.Start(myProcess.Id);
            }

            manualResetEventSlim.Set();
            Mutex.WaitAll(threadFinishEventsInternal.ToArray(), 60000);
            Console.WriteLine("Writing: " + count + " " + DateTime.Now.ToString());
            var c = 0;
            if (done)
            {
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
                wb.Save(_output + "dataset" + count + ".csv", SaveFormat.Csv);
                Console.WriteLine("All done: " + count);
            }
            else
            {
                Console.WriteLine("Failed: " + count);
            }
        }


        private static bool GetCounterData(PerformanceCounter counter, int index)
        {
            try
            {
                counter.NextValue();
                for (int i = 0; i < 60; i++)
                {
                    Thread.Sleep(500);
                    float val = counter.NextValue();
                    Data[index][counter.CounterName][i] = val;
                }

                return true;
            }
            catch (Exception e)
            {
                return false;
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
                try
                {
                    using PerformanceCounter cnt = new PerformanceCounter("Process",
                        "ID Process", instance, true);
                    int val = (int) cnt.RawValue;
                    if (val == processId)
                    {
                        return instance;
                    }
                }
                catch (Exception e)
                {
                    return null;
                }
            }

            return null;
        }
    }
}