import csv
import os
import time


class LocalLogger:
    def __init__(self, run_dir, writter):
        self.run_dir = run_dir
        self.writter = writter
        self.start_time = time.time()
        self._eval_file = None
        self._episode_file = None
        self._train_file = None
        self._train_writer = None

    def log_eval(self, total_steps, avg_return, avg_cost, avg_length):
        elapsed = time.time() - self.start_time
        print(
            f"[eval] steps={total_steps} return={avg_return:.2f} cost={avg_cost:.2f} "
            f"length={avg_length:.1f} elapsed={elapsed / 60:.1f}min",
            flush=True,
        )
        self.writter.add_scalar("eval/average_episode_return", avg_return, total_steps)
        self.writter.add_scalar("eval/average_episode_cost", avg_cost, total_steps)
        self.writter.add_scalar("eval/average_episode_length", avg_length, total_steps)
        if self._eval_file is None:
            self._eval_file = open(os.path.join(self.run_dir, "progress.csv"), "w", encoding="utf-8")
            self._eval_file.write("total_steps,eval_return,eval_cost,eval_length,elapsed_sec\n")
        self._eval_file.write(f"{total_steps},{avg_return},{avg_cost},{avg_length},{elapsed:.1f}\n")
        self._eval_file.flush()

    def log_eval_episodes(self, total_steps, returns, costs):
        if self._episode_file is None:
            self._episode_file = open(os.path.join(self.run_dir, "eval_episodes.csv"), "w", encoding="utf-8")
            self._episode_file.write("total_steps,episode,return,cost\n")
        for k, (ret, cost) in enumerate(zip(returns, costs)):
            self._episode_file.write(f"{total_steps},{k},{ret},{cost}\n")
        self._episode_file.flush()

    def log_train(self, total_steps, stats):
        for k, v in stats.items():
            self.writter.add_scalar(f"train/{k}", v, total_steps)
        if self._train_file is None:
            self._train_file = open(os.path.join(self.run_dir, "train_stats.csv"), "w", encoding="utf-8")
            self._train_writer = csv.DictWriter(
                self._train_file, fieldnames=["total_steps"] + list(stats.keys()), extrasaction="ignore", restval=""
            )
            self._train_writer.writeheader()
        self._train_writer.writerow({"total_steps": total_steps, **stats})
        self._train_file.flush()
        summary = " ".join(f"{k}={v:.3g}" for k, v in stats.items())
        print(f"[train] steps={total_steps} {summary}", flush=True)

    def close(self):
        for f in (self._eval_file, self._episode_file, self._train_file):
            if f is not None:
                f.close()
