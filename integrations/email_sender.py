"""
Модуль для отправки электронных писем.
Предоставляет функции для отправки уведомлений по электронной почте.
"""

import asyncio
import logging
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from typing import Dict, List, Any, Optional, Union

from project.config import get_config
from project.utils.logging_utils import get_logger
from project.utils.error_handler import async_handle_error

logger = get_logger(__name__)


class EmailSender:
    """
    Класс для отправки электронных писем.
    """

    _instance = None

    @classmethod
    def get_instance(cls) -> "EmailSender":
        """
        Получает экземпляр отправителя email (Singleton).

        Returns:
            Экземпляр класса EmailSender
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(
        self,
        smtp_server: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        Инициализирует отправителя электронных писем.

        Args:
            smtp_server: SMTP-сервер
            port: Порт SMTP-сервера
            username: Имя пользователя (email)
            password: Пароль
        """
        self.config = get_config()

        # В реальном приложении параметры должны быть в конфигурации
        # self.smtp_server = smtp_server or self.config.EMAIL_SMTP_SERVER
        # self.port = port or self.config.EMAIL_SMTP_PORT
        # self.username = username or self.config.EMAIL_USERNAME
        # self.password = password or self.config.EMAIL_PASSWORD

        # Для демонстрации используем значения по умолчанию
        self.smtp_server = smtp_server or "smtp.gmail.com"
        self.port = port or 587
        self.username = username
        self.password = password

        if not all([self.smtp_server, self.port, self.username, self.password]):
            logger.warning(
                "Email-отправитель не настроен - письма не будут отправляться"
            )
        else:
            logger.info(f"Email-отправитель инициализирован для {self.username}")

    @async_handle_error
    async def send_email(
        self,
        to_email: Union[str, List[str]],
        subject: str,
        body: str,
        html: Optional[str] = None,
        attachments: Optional[List[str]] = None,
    ) -> bool:
        """
        Отправляет электронное письмо.

        Args:
            to_email: Email получателя или список email-ов
            subject: Тема письма
            body: Текст письма
            html: HTML-версия письма
            attachments: Список путей к вложениям

        Returns:
            True при успешной отправке, иначе False
        """
        if not all([self.smtp_server, self.port, self.username, self.password]):
            logger.warning("Email-отправитель не настроен - письмо не отправлено")
            return False

        # Преобразуем один email в список
        if isinstance(to_email, str):
            to_email = [to_email]

        # Создаем сообщение
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.username
        msg["To"] = ", ".join(to_email)

        # Добавляем текстовую версию
        msg.attach(MIMEText(body, "plain"))

        # Добавляем HTML-версию, если она есть
        if html:
            msg.attach(MIMEText(html, "html"))

        # Добавляем вложения, если они есть
        if attachments:
            for file_path in attachments:
                try:
                    with open(file_path, "rb") as file:
                        part = MIMEApplication(
                            file.read(), Name=file_path.split("/")[-1]
                        )

                    part["Content-Disposition"] = (
                        f'attachment; filename="{file_path.split("/")[-1]}"'
                    )
                    msg.attach(part)
                except Exception as e:
                    logger.error(
                        f"Ошибка при добавлении вложения {file_path}: {str(e)}"
                    )

        # Запускаем отправку в отдельном потоке для неблокирующей работы
        return await asyncio.get_event_loop().run_in_executor(
            None, self._send_email_sync, to_email, msg
        )

    def _send_email_sync(self, to_email: List[str], msg: MIMEMultipart) -> bool:
        """
        Синхронная отправка электронного письма.

        Args:
            to_email: Список email-ов получателей
            msg: Подготовленное сообщение

        Returns:
            True при успешной отправке, иначе False
        """
        try:
            # Создаем защищенное соединение с SMTP-сервером
            context = ssl.create_default_context()

            with smtplib.SMTP(self.smtp_server, self.port) as server:
                server.starttls(context=context)
                server.login(self.username, self.password)
                server.sendmail(self.username, to_email, msg.as_string())

            logger.info(f"Письмо отправлено на {', '.join(to_email)}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при отправке письма: {str(e)}")
            return False

    @async_handle_error
    async def send_alert(
        self, to_email: Union[str, List[str]], title: str, message: str
    ) -> bool:
        """
        Отправляет предупреждающее сообщение по электронной почте.

        Args:
            to_email: Email получателя или список email-ов
            title: Заголовок сообщения
            message: Текст сообщения

        Returns:
            True при успешной отправке, иначе False
        """
        html = f"""
        <html>
            <head>
                <style>
                    .alert {{
                        padding: 15px;
                        border-radius: 4px;
                        color: #721c24;
                        background-color: #f8d7da;
                        border: 1px solid #f5c6cb;
                        margin-bottom: 20px;
                    }}
                    .alert-title {{
                        font-size: 18px;
                        font-weight: bold;
                        margin-bottom: 10px;
                    }}
                </style>
            </head>
            <body>
                <div class="alert">
                    <div class="alert-title">{title}</div>
                    <p>{message}</p>
                </div>
                <p>This is an automated alert from Trading Bot.</p>
            </body>
        </html>
        """

        return await self.send_email(
            to_email=to_email, subject=f"ALERT: {title}", body=message, html=html
        )

    @async_handle_error
    async def send_report(
        self,
        to_email: Union[str, List[str]],
        report_data: Dict[str, Any],
        report_type: str = "daily",
    ) -> bool:
        """
        Отправляет отчет по электронной почте.

        Args:
            to_email: Email получателя или список email-ов
            report_data: Данные для отчета
            report_type: Тип отчета (daily, weekly, monthly)

        Returns:
            True при успешной отправке, иначе False
        """
        # Формируем заголовок на основе типа отчета
        subjects = {
            "daily": "Daily Trading Report",
            "weekly": "Weekly Trading Report",
            "monthly": "Monthly Trading Report",
        }
        subject = subjects.get(report_type, "Trading Report")

        # Формируем текстовую версию отчета
        body = f"{subject}\n\n"

        for key, value in report_data.items():
            body += f"{key}: {value}\n"

        # Формируем HTML-версию отчета
        html = f"""
        <html>
            <head>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                    }}
                    .report {{
                        border: 1px solid #ddd;
                        border-radius: 4px;
                        padding: 20px;
                        margin-bottom: 20px;
                    }}
                    .report-title {{
                        font-size: 24px;
                        font-weight: bold;
                        margin-bottom: 20px;
                        color: #2c3e50;
                    }}
                    .report-item {{
                        margin-bottom: 10px;
                    }}
                    .report-item span {{
                        font-weight: bold;
                    }}
                    .positive {{
                        color: green;
                    }}
                    .negative {{
                        color: red;
                    }}
                </style>
            </head>
            <body>
                <div class="report">
                    <div class="report-title">{subject}</div>
        """

        for key, value in report_data.items():
            # Добавляем цвет для значений прибыли/убытка
            css_class = ""
            if "profit" in key.lower() or "return" in key.lower():
                if isinstance(value, (int, float)):
                    css_class = "positive" if value >= 0 else "negative"

            html += f"""
                    <div class="report-item">
                        <span>{key}:</span> <span class="{css_class}">{value}</span>
                    </div>
            """

        html += """
                </div>
                <p>This is an automated report from Trading Bot.</p>
            </body>
        </html>
        """

        # Отправляем отчет
        return await self.send_email(
            to_email=to_email, subject=subject, body=body, html=html
        )
