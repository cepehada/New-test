"""
Модуль отчетов.
Создает ежедневные CSV-отчеты, экспортирует данные в различные форматы 
и предоставляет гибкие возможности фильтрации и агрегации.
"""

import csv
import logging
import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
import pandas as pd
import aiofiles
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

from project.risk_management.portfolio_manager import PortfolioManager
from project.config import load_config

# Загрузка конфигурации
config = load_config()
logger = logging.getLogger("Reports")

class ReportsGenerator:
    """Класс для генерации различных типов отчетов."""
    
    def __init__(self, portfolio_manager: Optional[PortfolioManager] = None):
        """
        Инициализация генератора отчетов.
        
        Args:
            portfolio_manager: Экземпляр PortfolioManager. Если None, создается новый.
        """
        self.pm = portfolio_manager or PortfolioManager()
        self.reports_dir = Path(config.get("reports_path", "reports"))
        
        # Создаем директорию для отчетов, если она не существует
        os.makedirs(self.reports_dir, exist_ok=True)
    
    async def get_trades_data(
        self, 
        start_date: Optional[datetime] = None, 
        end_date: Optional[datetime] = None,
        symbols: Optional[List[str]] = None,
        min_profit: Optional[float] = None,
        max_profit: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Получает данные о сделках с возможностью фильтрации.
        
        Args:
            start_date: Начальная дата для фильтрации
            end_date: Конечная дата для фильтрации
            symbols: Список торговых пар для фильтрации
            min_profit: Минимальный профит для фильтрации
            max_profit: Максимальный профит для фильтрации
            
        Returns:
            Отфильтрованный список сделок
        """
        trades = await self.pm.get_closed_trades()
        filtered_trades = []
        
        for trade in trades:
            # Преобразование строки времени в объект datetime
            trade_time = datetime.fromisoformat(trade["timestamp"].replace('Z', '+00:00'))
            
            # Применение фильтров
            if start_date and trade_time < start_date:
                continue
            if end_date and trade_time > end_date:
                continue
            if symbols and trade["symbol"] not in symbols:
                continue
            if min_profit is not None and trade["pnl"] < min_profit:
                continue
            if max_profit is not None and trade["pnl"] > max_profit:
                continue
                
            filtered_trades.append(trade)
        
        return filtered_trades
    
    async def generate_daily_csv_report(
        self, 
        file_path: Optional[str] = None,
        date: Optional[datetime] = None,
        additional_fields: Optional[List[str]] = None
    ) -> str:
        """
        Генерирует ежедневный отчет в формате CSV.

        Args:
            file_path: Путь для сохранения отчета. Если None, генерируется автоматически.
            date: Дата отчета. Если None, используется текущая дата.
            additional_fields: Дополнительные поля для включения в отчет.

        Returns:
            Путь к созданному файлу отчета.
        """
        try:
            # Определение даты отчета
            report_date = date or datetime.now()
            date_str = report_date.strftime("%Y-%m-%d")
            
            # Автоматическое формирование имени файла, если не указано
            if not file_path:
                file_path = self.reports_dir / f"daily_report_{date_str}.csv"
            
            # Получение данных за указанную дату
            start_date = datetime(report_date.year, report_date.month, report_date.day)
            end_date = start_date + timedelta(days=1)
            trades = await self.get_trades_data(start_date=start_date, end_date=end_date)
            
            # Определение полей для отчета
            default_fields = ["id", "symbol", "pnl", "timestamp", "side", "entry_price", "exit_price"]
            fields = default_fields + (additional_fields or [])
            
            # Добавление метаданных
            metadata = {
                "report_type": "Daily Trading Report",
                "date": date_str,
                "generated_at": datetime.now().isoformat(),
                "total_trades": len(trades),
                "total_pnl": sum(trade.get("pnl", 0) for trade in trades)
            }
            
            # Запись данных с использованием aiofiles для асинхронной записи
            async with aiofiles.open(file_path, "w", newline="", encoding="utf-8") as f:
                # Запись метаданных в виде комментариев
                await f.write(f"# {json.dumps(metadata)}\n")
                
                # Создание объекта writer и запись заголовка
                writer = csv.writer(await f.drain())
                writer.writerow(fields)
                
                # Запись данных сделок
                for trade in trades:
                    row = [trade.get(field, "") for field in fields]
                    writer.writerow(row)
            
            logger.info(f"Ежедневный отчет сохранен: {file_path}")
            return str(file_path)
        
        except Exception as e:
            logger.error(f"Ошибка при создании ежедневного отчета: {e}", exc_info=True)
            raise
    
    async def generate_summary_report(
        self,
        period: str = "weekly",
        file_format: str = "csv",
        output_path: Optional[str] = None
    ) -> str:
        """
        Генерирует сводный отчет за указанный период.
        
        Args:
            period: Период отчета ('daily', 'weekly', 'monthly', 'quarterly', 'yearly')
            file_format: Формат файла ('csv', 'json', 'xlsx', 'html')
            output_path: Путь для сохранения. Если None, генерируется автоматически.
            
        Returns:
            Путь к созданному файлу отчета.
        """
        try:
            # Определение начальной и конечной даты на основе периода
            end_date = datetime.now()
            
            if period == "daily":
                start_date = datetime(end_date.year, end_date.month, end_date.day)
                period_str = end_date.strftime("%Y-%m-%d")
            elif period == "weekly":
                start_date = end_date - timedelta(days=end_date.weekday(), weeks=0)
                period_str = f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
            elif period == "monthly":
                start_date = datetime(end_date.year, end_date.month, 1)
                period_str = end_date.strftime("%Y-%m")
            elif period == "quarterly":
                quarter = (end_date.month - 1) // 3 + 1
                start_date = datetime(end_date.year, (quarter - 1) * 3 + 1, 1)
                period_str = f"{end_date.year}_Q{quarter}"
            elif period == "yearly":
                start_date = datetime(end_date.year, 1, 1)
                period_str = str(end_date.year)
            else:
                raise ValueError(f"Неподдерживаемый период: {period}")
            
            # Автоматическое формирование имени файла, если не указано
            if not output_path:
                output_path = self.reports_dir / f"summary_report_{period}_{period_str}.{file_format}"
            
            # Получение данных за указанный период
            trades = await self.get_trades_data(start_date=start_date, end_date=end_date)
            
            # Создание DataFrame для агрегации и анализа
            df = pd.DataFrame(trades)
            
            if len(df) == 0:
                logger.warning(f"Нет данных для отчета за период {period_str}")
                # Создаем пустой отчет
                df = pd.DataFrame(columns=["symbol", "pnl", "count", "avg_profit", "win_rate"])
            else:
                # Преобразование данных
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                
                # Рассчет агрегированных метрик по символам
                summary = df.groupby("symbol").agg(
                    pnl=("pnl", "sum"),
                    count=("id", "count"),
                    avg_profit=("pnl", "mean"),
                    win_rate=("pnl", lambda x: (x > 0).mean() * 100)
                ).reset_index()
                
                # Сортировка по прибыли
                summary = summary.sort_values("pnl", ascending=False)
                
                # Добавление общей строки
                total_row = pd.DataFrame([{
                    "symbol": "TOTAL",
                    "pnl": summary["pnl"].sum(),
                    "count": summary["count"].sum(),
                    "avg_profit": df["pnl"].mean(),
                    "win_rate": (df["pnl"] > 0).mean() * 100
                }])
                
                df = pd.concat([summary, total_row], ignore_index=True)
            
            # Метаданные для отчета
            metadata = {
                "report_type": f"{period.capitalize()} Summary Report",
                "period": period,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "generated_at": datetime.now().isoformat(),
                "total_trades": len(trades),
                "total_pnl": sum(trade.get("pnl", 0) for trade in trades)
            }
            
            # Сохранение отчета в указанный формат
            if file_format == "csv":
                with open(output_path, "w", newline="", encoding="utf-8") as f:
                    f.write(f"# {json.dumps(metadata)}\n")
                    df.to_csv(f, index=False)
            elif file_format == "json":
                result = {
                    "metadata": metadata,
                    "data": df.to_dict(orient="records")
                }
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
            elif file_format == "xlsx":
                with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                    df.to_excel(writer, sheet_name="Summary", index=False)
                    # Добавление листа с метаданными
                    pd.DataFrame([metadata]).to_excel(writer, sheet_name="Metadata", index=False)
            elif file_format == "html":
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>{metadata['report_type']}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                        .metadata {{ margin-bottom: 20px; }}
                    </style>
                </head>
                <body>
                    <div class="metadata">
                        <h2>{metadata['report_type']}</h2>
                        <p>Период: {metadata['start_date']} - {metadata['end_date']}</p>
                        <p>Всего сделок: {metadata['total_trades']}</p>
                        <p>Общий P&L: {metadata['total_pnl']:.2f}</p>
                        <p>Сгенерировано: {metadata['generated_at']}</p>
                    </div>
                    {df.to_html(index=False)}
                </body>
                </html>
                """
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
            else:
                raise ValueError(f"Неподдерживаемый формат файла: {file_format}")
            
            logger.info(f"Сводный отчет сохранен: {output_path}")
            return str(output_path)
        
        except Exception as e:
            logger.error(f"Ошибка при создании сводного отчета: {e}", exc_info=True)
            raise
    
    async def compliance_log_backup(
        self, 
        log_file: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> str:
        """
        Создает резервную копию журнала сделок для аудита.

        Args:
            log_file: Путь к файлу для журнала. Если None, генерируется автоматически.
            start_date: Начальная дата для фильтрации
            end_date: Конечная дата для фильтрации

        Returns:
            Путь к созданному файлу журнала.
        """
        try:
            # Автоматическое формирование имени файла, если не указано
            if not log_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = self.reports_dir / f"compliance_log_{timestamp}.json"
            
            # Получение и фильтрация данных
            trades = await self.get_trades_data(start_date=start_date, end_date=end_date)
            
            # Добавление метаданных
            log_data = {
                "metadata": {
                    "backup_time": datetime.now().isoformat(),
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None,
                    "total_trades": len(trades)
                },
                "trades": trades
            }
            
            # Запись данных
            async with aiofiles.open(log_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(log_data, indent=2))
            
            logger.info(f"Журнал сделок сохранен: {log_file}")
            return str(log_file)
        
        except Exception as e:
            logger.error(f"Ошибка при создании резервной копии журнала: {e}", exc_info=True)
            raise
    
    async def send_report_by_email(
        self,
        file_path: str,
        recipients: List[str],
        subject: str = "Торговый отчет",
        body: str = "Во вложении находится торговый отчет."
    ) -> bool:
        """
        Отправляет отчет по электронной почте.
        
        Args:
            file_path: Путь к файлу отчета
            recipients: Список адресов электронной почты получателей
            subject: Тема письма
            body: Текст письма
            
        Returns:
            True в случае успешной отправки, иначе False
        """
        try:
            # Получение конфигурации SMTP из настроек
            smtp_config = config.get("smtp", {})
            smtp_server = smtp_config.get("server", "smtp.example.com")
            smtp_port = smtp_config.get("port", 587)
            smtp_user = smtp_config.get("user", "user@example.com")
            smtp_password = smtp_config.get("password", "password")
            sender = smtp_config.get("sender", "reports@trading-system.com")
            
            # Создание сообщения
            msg = MIMEMultipart()
            msg["From"] = sender
            msg["To"] = ", ".join(recipients)
            msg["Subject"] = subject
            
            # Добавление текста сообщения
            msg.attach(MIMEText(body, "plain"))
            
            # Чтение и добавление файла отчета
            with open(file_path, "rb") as f:
                attachment = MIMEApplication(f.read(), Name=os.path.basename(file_path))
                attachment["Content-Disposition"] = f'attachment; filename="{os.path.basename(file_path)}"'
                msg.attach(attachment)
            
            # Отправка сообщения
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
            
            logger.info(f"Отчет отправлен по электронной почте получателям: {', '.join(recipients)}")
            return True
        
        except Exception as e:
            logger.error(f"Ошибка при отправке отчета по электронной почте: {e}", exc_info=True)
            return False


async def scheduled_daily_report():
    """
    Функция для запуска по расписанию для создания и отправки ежедневного отчета.
    """
    try:
        generator = ReportsGenerator()
        
        # Генерация отчета за предыдущий день
        yesterday = datetime.now() - timedelta(days=1)
        
        # CSV отчет
        csv_path = await generator.generate_daily_csv_report(date=yesterday)
        
        # Сводный отчет в HTML
        html_path = await generator.generate_summary_report(period="daily", file_format="html")
        
        # Отправка отчетов по электронной почте
        recipients = config.get("report_recipients", ["admin@example.com"])
        subject = f"Ежедневный торговый отчет за {yesterday.strftime('%Y-%m-%d')}"
        
        await generator.send_report_by_email(
            csv_path, 
            recipients,
            subject=subject,
            body="Во вложении находятся ежедневные торговые отчеты."
        )
        
        await generator.send_report_by_email(
            html_path, 
            recipients,
            subject=f"{subject} (HTML)",
            body="Во вложении находится сводный HTML-отчет."
        )
        
        # Создание резервной копии журнала
        await generator.compliance_log_backup(
            start_date=datetime(yesterday.year, yesterday.month, yesterday.day),
            end_date=datetime(yesterday.year, yesterday.month, yesterday.day) + timedelta(days=1)
        )
        
        logger.info("Запланированная задача по созданию отчетов выполнена успешно")
    
    except Exception as e:
        logger.error(f"Ошибка при выполнении запланированной задачи по созданию отчетов: {e}", exc_info=True)


if __name__ == "__main__":
    # Пример использования
    async def test_reports():
        generator = ReportsGenerator()
        
        # Генерация ежедневного отчета
        csv_path = await generator.generate_daily_csv_report()
        print(f"CSV отчет создан: {csv_path}")
        
        # Генерация сводного отчета за неделю в JSON
        json_path = await generator.generate_summary_report(period="weekly", file_format="json")
        print(f"JSON отчет создан: {json_path}")
        
        # Резервное копирование журнала
        log_path = await generator.compliance_log_backup()
        print(f"Журнал сохранен: {log_path}")
    
    asyncio.run(test_reports())