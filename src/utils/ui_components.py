"""
UI Components Library

Provides reusable UI components for configuration panels.
All components follow a consistent interface and design pattern.

Design Principles:
- Consistent API: All components implement the same interface
- Type Safety: Strong type hints and value validation
- Visual Feedback: Clear visual states (hover, active, disabled)
- Accessibility: Clear labels and value displays
- Extensibility: Easy to add new component types

Author: MARL Platform Team
Version: 1.0.0
"""

import pygame
from typing import Any, Optional, Dict, List, Callable, Tuple
from abc import ABC, abstractmethod
from enum import Enum
import math

from src.config.ui_config import COLORS, FONT_SIZES


class ComponentState(Enum):
    """Component visual state"""
    NORMAL = "normal"
    HOVER = "hover"
    ACTIVE = "active"
    DISABLED = "disabled"


class UIComponent(ABC):
    """
    Base class for all UI components
    
    Provides common functionality:
    - Drawing interface
    - Event handling
    - Value management
    - State management
    """
    
    def __init__(
        self,
        rect: pygame.Rect,
        label: str = "",
        font_manager: Optional[Any] = None,
        enabled: bool = True
    ):
        """
        Initialize UI component
        
        Args:
            rect: Component rectangle (position and size)
            label: Component label text
            font_manager: Font manager for text rendering
            enabled: Whether component is enabled
        """
        self.rect = rect
        self.label = label
        self.font_manager = font_manager
        self.enabled = enabled
        self.state = ComponentState.NORMAL
        self._hover = False
        
        # Default font if no font_manager provided
        if font_manager is None:
            pygame.font.init()
            self._default_font = pygame.font.Font(None, FONT_SIZES.get('SMALL', 14))
        else:
            self._default_font = None
    
    def get_font(self, size_name: str = 'SMALL') -> pygame.font.Font:
        """Get font for rendering"""
        if self.font_manager:
            return self.font_manager.get_font(size_name)
        return self._default_font
    
    def render_text(self, text: str, size_name: str = 'SMALL', color: Tuple = COLORS['TEXT_PRIMARY']) -> pygame.Surface:
        """Render text"""
        if self.font_manager:
            return self.font_manager.render_text(text, size_name, color)
        font = self.get_font(size_name)
        return font.render(str(text), True, color)
    
    def is_point_inside(self, point: Tuple[int, int]) -> bool:
        """Check if point is inside component"""
        return self.rect.collidepoint(point) and self.enabled
    
    @abstractmethod
    def draw(self, screen: pygame.Surface) -> None:
        """Draw component on screen"""
        pass
    
    @abstractmethod
    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle pygame event
        
        Args:
            event: Pygame event
            
        Returns:
            True if event was handled, False otherwise
        """
        pass
    
    @abstractmethod
    def get_value(self) -> Any:
        """Get component value"""
        pass
    
    @abstractmethod
    def set_value(self, value: Any) -> None:
        """Set component value"""
        pass
    
    def update_hover(self, mouse_pos: Tuple[int, int]) -> None:
        """Update hover state based on mouse position"""
        self._hover = self.is_point_inside(mouse_pos)
        if self._hover and self.enabled:
            self.state = ComponentState.HOVER
        elif not self.enabled:
            self.state = ComponentState.DISABLED
        else:
            self.state = ComponentState.NORMAL


class Slider(UIComponent):
    """
    Slider component for numeric input
    
    Features:
    - Draggable handle
    - Value display
    - Min/max labels
    - Step size support
    """
    
    def __init__(
        self,
        rect: pygame.Rect,
        label: str = "",
        min_value: float = 0.0,
        max_value: float = 100.0,
        initial_value: float = 50.0,
        step: float = 1.0,
        font_manager: Optional[Any] = None,
        enabled: bool = True,
        value_format: str = "{:.2f}",
        show_value: bool = True
    ):
        """
        Initialize slider
        
        Args:
            rect: Component rectangle
            label: Label text
            min_value: Minimum value
            max_value: Maximum value
            initial_value: Initial value
            step: Step size for value changes
            font_manager: Font manager
            enabled: Whether enabled
            value_format: Format string for value display
            show_value: Whether to show current value
        """
        super().__init__(rect, label, font_manager, enabled)
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.value = max(min_value, min(max_value, initial_value))
        self.value_format = value_format
        self.show_value = show_value
        
        # Handle dimensions
        self.handle_width = 12
        self.handle_height = 20
        self.track_height = 4
        self.track_y = self.rect.centery - self.track_height // 2
        
        # Dragging state
        self._dragging = False
        self._drag_offset = 0
    
    def _value_to_position(self, value: float) -> float:
        """Convert value to x position"""
        normalized = (value - self.min_value) / (self.max_value - self.min_value)
        return self.rect.left + normalized * (self.rect.width - self.handle_width)
    
    def _position_to_value(self, x: float) -> float:
        """Convert x position to value"""
        normalized = (x - self.rect.left - self.handle_width / 2) / (self.rect.width - self.handle_width)
        normalized = max(0.0, min(1.0, normalized))
        value = self.min_value + normalized * (self.max_value - self.min_value)
        # Snap to step
        if self.step > 0:
            value = round(value / self.step) * self.step
        return max(self.min_value, min(self.max_value, value))
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draw slider"""
        # Background track
        track_rect = pygame.Rect(
            self.rect.left,
            self.track_y,
            self.rect.width,
            self.track_height
        )
        track_color = COLORS.get('LIGHT_GRAY', (200, 200, 200)) if self.enabled else COLORS.get('GRAY', (128, 128, 128))
        pygame.draw.rect(screen, track_color, track_rect)
        pygame.draw.rect(screen, COLORS.get('PANEL_BORDER', (200, 200, 200)), track_rect, 1)
        
        # Filled portion
        handle_x = self._value_to_position(self.value)
        filled_width = handle_x - self.rect.left + self.handle_width / 2
        if filled_width > 0:
            filled_rect = pygame.Rect(
                self.rect.left,
                self.track_y,
                int(filled_width),
                self.track_height
            )
            fill_color = COLORS.get('INFO', (0, 123, 255)) if self.enabled else COLORS.get('GRAY', (128, 128, 128))
            pygame.draw.rect(screen, fill_color, filled_rect)
        
        # Handle
        handle_rect = pygame.Rect(
            int(handle_x),
            self.rect.centery - self.handle_height // 2,
            self.handle_width,
            self.handle_height
        )
        
        # Handle color based on state
        if not self.enabled:
            handle_color = COLORS.get('GRAY', (128, 128, 128))
        elif self._dragging or self.state == ComponentState.HOVER:
            handle_color = COLORS.get('BUTTON_ACTIVE', (0, 123, 255))
        else:
            handle_color = COLORS.get('INFO', (0, 123, 255))
        
        pygame.draw.rect(screen, handle_color, handle_rect)
        pygame.draw.rect(screen, COLORS.get('PANEL_BORDER', (150, 150, 150)), handle_rect, 2)
        
        # Label and value
        if self.label:
            label_surface = self.render_text(self.label, 'TINY', COLORS['TEXT_PRIMARY'])
            screen.blit(label_surface, (self.rect.left, self.rect.top - 16))
        
        if self.show_value:
            value_text = self.value_format.format(self.value)
            value_surface = self.render_text(value_text, 'TINY', COLORS['TEXT_PRIMARY'])
            value_x = self.rect.right - value_surface.get_width()
            screen.blit(value_surface, (value_x, self.rect.top - 16))
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle events"""
        if not self.enabled:
            return False
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                mouse_pos = event.pos
                handle_x = self._value_to_position(self.value)
                handle_rect = pygame.Rect(
                    int(handle_x),
                    self.rect.centery - self.handle_height // 2,
                    self.handle_width,
                    self.handle_height
                )
                
                if handle_rect.collidepoint(mouse_pos):
                    # Start dragging
                    self._dragging = True
                    self._drag_offset = mouse_pos[0] - handle_x
                    self.state = ComponentState.ACTIVE
                    return True
                elif self.rect.collidepoint(mouse_pos):
                    # Click on track - jump to position
                    self.value = self._position_to_value(mouse_pos[0])
                    return True
        
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                if self._dragging:
                    self._dragging = False
                    self.state = ComponentState.NORMAL
                    return True
        
        elif event.type == pygame.MOUSEMOTION:
            if self._dragging:
                new_value = self._position_to_value(event.pos[0] - self._drag_offset)
                if new_value != self.value:
                    self.value = new_value
                    return True
        
        return False
    
    def get_value(self) -> float:
        """Get current value"""
        return self.value
    
    def set_value(self, value: float) -> None:
        """Set value"""
        self.value = max(self.min_value, min(self.max_value, value))
        if self.step > 0:
            self.value = round(self.value / self.step) * self.step


class InputBox(UIComponent):
    """
    Text input box component
    
    Features:
    - Text input
    - Numeric validation
    - Focus state
    - Value formatting
    """
    
    def __init__(
        self,
        rect: pygame.Rect,
        label: str = "",
        initial_value: str = "",
        numeric: bool = False,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        font_manager: Optional[Any] = None,
        enabled: bool = True
    ):
        """
        Initialize input box
        
        Args:
            rect: Component rectangle
            label: Label text
            initial_value: Initial text value
            numeric: Whether to validate as numeric
            min_value: Minimum value (if numeric)
            max_value: Maximum value (if numeric)
            font_manager: Font manager
            enabled: Whether enabled
        """
        super().__init__(rect, label, font_manager, enabled)
        self.value = str(initial_value)
        self.numeric = numeric
        self.min_value = min_value
        self.max_value = max_value
        self._focused = False
        self._cursor_pos = len(self.value)
        self._cursor_timer = 0
    
    def _validate_value(self, value: str) -> bool:
        """Validate input value"""
        if not value:
            return True  # Empty is allowed
        
        if self.numeric:
            try:
                num_value = float(value)
                if self.min_value is not None and num_value < self.min_value:
                    return False
                if self.max_value is not None and num_value > self.max_value:
                    return False
            except ValueError:
                return False
        
        return True
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draw input box"""
        # Background
        bg_color = COLORS.get('PANEL_BG', (255, 255, 255)) if self.enabled else COLORS.get('LIGHT_GRAY', (220, 220, 220))
        border_color = COLORS.get('INFO', (0, 123, 255)) if self._focused else COLORS.get('PANEL_BORDER', (200, 200, 200))
        
        if not self.enabled:
            bg_color = COLORS.get('LIGHT_GRAY', (220, 220, 220))
            border_color = COLORS.get('GRAY', (128, 128, 128))
        
        pygame.draw.rect(screen, bg_color, self.rect)
        pygame.draw.rect(screen, border_color, self.rect, 2)
        
        # Text
        if self.value:
            text_surface = self.render_text(self.value, 'SMALL', COLORS['TEXT_PRIMARY'])
            text_y = self.rect.centery - text_surface.get_height() // 2
            screen.blit(text_surface, (self.rect.left + 4, text_y))
        
        # Cursor (when focused)
        if self._focused and self.enabled:
            self._cursor_timer += 1
            if (self._cursor_timer // 30) % 2 == 0:  # Blink cursor
                cursor_x = self.rect.left + 4
                if self.value:
                    text_surface = self.render_text(self.value[:self._cursor_pos], 'SMALL', COLORS['TEXT_PRIMARY'])
                    cursor_x += text_surface.get_width()
                cursor_y = self.rect.centery - 8
                pygame.draw.line(screen, COLORS['TEXT_PRIMARY'], (cursor_x, cursor_y), (cursor_x, cursor_y + 16), 2)
        
        # Label
        if self.label:
            label_surface = self.render_text(self.label, 'TINY', COLORS['TEXT_PRIMARY'])
            screen.blit(label_surface, (self.rect.left, self.rect.top - 16))
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle events"""
        if not self.enabled:
            return False
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if self.rect.collidepoint(event.pos):
                    self._focused = True
                    self._cursor_timer = 0
                    return True
                else:
                    self._focused = False
                    return False
        
        if not self._focused:
            return False
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                if self._cursor_pos > 0:
                    new_value = self.value[:self._cursor_pos - 1] + self.value[self._cursor_pos:]
                    if self._validate_value(new_value):
                        self.value = new_value
                        self._cursor_pos -= 1
                    return True
            elif event.key == pygame.K_DELETE:
                if self._cursor_pos < len(self.value):
                    new_value = self.value[:self._cursor_pos] + self.value[self._cursor_pos + 1:]
                    if self._validate_value(new_value):
                        self.value = new_value
                    return True
            elif event.key == pygame.K_LEFT:
                self._cursor_pos = max(0, self._cursor_pos - 1)
                return True
            elif event.key == pygame.K_RIGHT:
                self._cursor_pos = min(len(self.value), self._cursor_pos + 1)
                return True
            elif event.key == pygame.K_HOME:
                self._cursor_pos = 0
                return True
            elif event.key == pygame.K_END:
                self._cursor_pos = len(self.value)
                return True
            elif event.unicode and event.unicode.isprintable():
                new_value = self.value[:self._cursor_pos] + event.unicode + self.value[self._cursor_pos:]
                if self._validate_value(new_value):
                    self.value = new_value
                    self._cursor_pos += 1
                return True
        
        return False
    
    def get_value(self) -> str:
        """Get current value"""
        return self.value
    
    def set_value(self, value: Any) -> None:
        """Set value"""
        self.value = str(value)
        self._cursor_pos = len(self.value)
    
    def get_numeric_value(self) -> Optional[float]:
        """Get numeric value if numeric mode"""
        if self.numeric:
            try:
                return float(self.value)
            except ValueError:
                return None
        return None


class Dropdown(UIComponent):
    """
    Dropdown menu component
    
    Features:
    - Option selection
    - Dropdown list
    - Search/filter support (optional)
    """
    
    def __init__(
        self,
        rect: pygame.Rect,
        label: str = "",
        options: List[str] = None,
        initial_value: Optional[str] = None,
        font_manager: Optional[Any] = None,
        enabled: bool = True
    ):
        """
        Initialize dropdown
        
        Args:
            rect: Component rectangle
            label: Label text
            options: List of option strings
            initial_value: Initial selected value
            font_manager: Font manager
            enabled: Whether enabled
        """
        super().__init__(rect, label, font_manager, enabled)
        self.options = options or []
        self.selected_value = initial_value if initial_value in self.options else (self.options[0] if self.options else None)
        self._expanded = False
        self._dropdown_height = min(150, len(self.options) * 25 + 10)
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draw dropdown"""
        # Main box
        bg_color = COLORS.get('PANEL_BG', (255, 255, 255)) if self.enabled else COLORS.get('LIGHT_GRAY', (220, 220, 220))
        border_color = COLORS.get('INFO', (0, 123, 255)) if self._expanded else COLORS.get('PANEL_BORDER', (200, 200, 200))
        
        if not self.enabled:
            bg_color = COLORS.get('LIGHT_GRAY', (220, 220, 220))
            border_color = COLORS.get('GRAY', (128, 128, 128))
        
        pygame.draw.rect(screen, bg_color, self.rect)
        pygame.draw.rect(screen, border_color, self.rect, 2)
        
        # Selected value
        if self.selected_value:
            text_surface = self.render_text(self.selected_value, 'SMALL', COLORS['TEXT_PRIMARY'])
            text_y = self.rect.centery - text_surface.get_height() // 2
            screen.blit(text_surface, (self.rect.left + 4, text_y))
        
        # Dropdown arrow
        arrow_size = 8
        arrow_x = self.rect.right - arrow_size - 4
        arrow_y = self.rect.centery
        arrow_points = [
            (arrow_x, arrow_y - arrow_size // 2),
            (arrow_x + arrow_size, arrow_y - arrow_size // 2),
            (arrow_x + arrow_size // 2, arrow_y + arrow_size // 2)
        ]
        pygame.draw.polygon(screen, COLORS['TEXT_PRIMARY'], arrow_points)
        
        # Dropdown list
        if self._expanded and self.enabled:
            dropdown_rect = pygame.Rect(
                self.rect.left,
                self.rect.bottom,
                self.rect.width,
                self._dropdown_height
            )
            pygame.draw.rect(screen, COLORS.get('CHART_BG', (250, 250, 250)), dropdown_rect)
            pygame.draw.rect(screen, COLORS.get('PANEL_BORDER', (150, 150, 150)), dropdown_rect, 2)
            
            item_height = 25
            start_y = dropdown_rect.top + 5
            for i, option in enumerate(self.options):
                item_rect = pygame.Rect(
                    dropdown_rect.left + 2,
                    start_y + i * item_height,
                    dropdown_rect.width - 4,
                    item_height - 2
                )
                
                # Highlight selected and hover
                if option == self.selected_value:
                    pygame.draw.rect(screen, COLORS.get('INFO', (0, 123, 255)), item_rect)
                elif item_rect.collidepoint(pygame.mouse.get_pos()):
                    pygame.draw.rect(screen, COLORS.get('BUTTON_HOVER', (230, 230, 230)), item_rect)
                
                option_surface = self.render_text(option, 'TINY', COLORS['TEXT_PRIMARY'])
                screen.blit(option_surface, (item_rect.left + 4, item_rect.centery - option_surface.get_height() // 2))
        
        # Label
        if self.label:
            label_surface = self.render_text(self.label, 'TINY', COLORS['TEXT_PRIMARY'])
            screen.blit(label_surface, (self.rect.left, self.rect.top - 16))
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle events"""
        if not self.enabled:
            return False
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_pos = event.pos
                
                if self.rect.collidepoint(mouse_pos):
                    # Toggle dropdown
                    self._expanded = not self._expanded
                    return True
                
                # Check dropdown items
                if self._expanded:
                    dropdown_rect = pygame.Rect(
                        self.rect.left,
                        self.rect.bottom,
                        self.rect.width,
                        self._dropdown_height
                    )
                    
                    if dropdown_rect.collidepoint(mouse_pos):
                        item_height = 25
                        item_index = int((mouse_pos[1] - dropdown_rect.top - 5) / item_height)
                        if 0 <= item_index < len(self.options):
                            self.selected_value = self.options[item_index]
                            self._expanded = False
                            return True
                    else:
                        # Click outside - close dropdown
                        self._expanded = False
                        return False
        
        return False
    
    def get_value(self) -> Optional[str]:
        """Get selected value"""
        return self.selected_value
    
    def set_value(self, value: str) -> None:
        """Set selected value"""
        if value in self.options:
            self.selected_value = value


class Checkbox(UIComponent):
    """
    Checkbox component
    
    Features:
    - Boolean state
    - Visual checkmark
    - Label support
    """
    
    def __init__(
        self,
        rect: pygame.Rect,
        label: str = "",
        initial_value: bool = False,
        font_manager: Optional[Any] = None,
        enabled: bool = True
    ):
        """
        Initialize checkbox
        
        Args:
            rect: Component rectangle
            label: Label text
            initial_value: Initial checked state
            font_manager: Font manager
            enabled: Whether enabled
        """
        super().__init__(rect, label, font_manager, enabled)
        self.checked = initial_value
        self.box_size = min(rect.width, rect.height, 20)
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draw checkbox"""
        # Box
        box_rect = pygame.Rect(
            self.rect.left,
            self.rect.centery - self.box_size // 2,
            self.box_size,
            self.box_size
        )
        
        bg_color = COLORS.get('PANEL_BG', (255, 255, 255)) if self.enabled else COLORS.get('LIGHT_GRAY', (220, 220, 220))
        border_color = COLORS.get('INFO', (0, 123, 255)) if self.checked else COLORS.get('PANEL_BORDER', (200, 200, 200))
        
        if not self.enabled:
            bg_color = COLORS.get('LIGHT_GRAY', (220, 220, 220))
            border_color = COLORS.get('GRAY', (128, 128, 128))
        
        pygame.draw.rect(screen, bg_color, box_rect)
        pygame.draw.rect(screen, border_color, box_rect, 2)
        
        # Checkmark
        if self.checked:
            check_color = COLORS.get('TEXT_PRIMARY', (0, 0, 0)) if self.enabled else COLORS.get('GRAY', (120, 120, 120))
            # Draw checkmark (simple X pattern)
            margin = 4
            pygame.draw.line(
                screen, check_color,
                (box_rect.left + margin, box_rect.centery),
                (box_rect.left + box_rect.width // 3, box_rect.bottom - margin),
                3
            )
            pygame.draw.line(
                screen, check_color,
                (box_rect.left + box_rect.width // 3, box_rect.bottom - margin),
                (box_rect.right - margin, box_rect.top + margin),
                3
            )
        
        # Label
        if self.label:
            label_surface = self.render_text(self.label, 'SMALL', COLORS['TEXT_PRIMARY'])
            label_x = box_rect.right + 8
            screen.blit(label_surface, (label_x, self.rect.centery - label_surface.get_height() // 2))
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle events"""
        if not self.enabled:
            return False
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if self.rect.collidepoint(event.pos):
                    self.checked = not self.checked
                    return True
        
        return False
    
    def get_value(self) -> bool:
        """Get checked state"""
        return self.checked
    
    def set_value(self, value: bool) -> None:
        """Set checked state"""
        self.checked = bool(value)


class CollapsiblePanel(UIComponent):
    """
    Collapsible Panel Component
    
    A panel that can be expanded or collapsed by clicking on the header.
    Contains child components that are shown/hidden based on the panel state.
    """
    
    def __init__(
        self,
        rect: pygame.Rect,
        title: str,
        font_manager: Optional[Any] = None,
        initial_expanded: bool = True,
        header_height: int = 35,
        child_components: Optional[List[UIComponent]] = None
    ):
        """
        Initialize collapsible panel
        
        Args:
            rect: Panel rectangle
            title: Panel title
            font_manager: Font manager for text rendering
            initial_expanded: Whether panel starts expanded
            header_height: Height of the header bar
            child_components: List of child components to manage
        """
        super().__init__(rect, title, font_manager)
        self.title = title
        self.is_expanded = initial_expanded
        self.header_height = header_height
        self.child_components = child_components or []
        
        # Header rectangle
        self.header_rect = pygame.Rect(
            rect.x, rect.y, rect.width, header_height
        )
        
        # Content rectangle (for child components)
        self.content_rect = pygame.Rect(
            rect.x, rect.y + header_height, rect.width, rect.height - header_height
        )
        
        # Animation state (for smooth expand/collapse)
        self.target_height = rect.height if initial_expanded else header_height
        self.current_height = self.target_height
        self.animation_speed = 10  # pixels per frame
        
        # Visual state
        self.is_hover = False
    
    def add_child(self, component: UIComponent) -> None:
        """Add a child component to the panel"""
        self.child_components.append(component)
    
    def remove_child(self, component: UIComponent) -> None:
        """Remove a child component from the panel"""
        if component in self.child_components:
            self.child_components.remove(component)
    
    def toggle(self) -> None:
        """Toggle expanded/collapsed state"""
        self.is_expanded = not self.is_expanded
        if self.is_expanded:
            self.target_height = self.rect.height
        else:
            self.target_height = self.header_height
    
    def set_expanded(self, expanded: bool) -> None:
        """Set expanded state"""
        if self.is_expanded != expanded:
            self.toggle()
    
    def get_value(self) -> bool:
        """Get current expanded state"""
        return self.is_expanded
    
    def set_value(self, value: bool) -> None:
        """Set expanded state"""
        self.set_expanded(value)
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draw collapsible panel"""
        # Update animation
        if abs(self.current_height - self.target_height) > 1:
            diff = self.target_height - self.current_height
            self.current_height += math.copysign(
                min(abs(diff), self.animation_speed), diff
            )
        else:
            self.current_height = self.target_height
        
        # Update content visibility based on expansion
        content_visible = self.is_expanded and self.current_height > self.header_height
        
        # Draw header
        header_color = COLORS.get('BUTTON_HOVER', (200, 200, 200)) if self.is_hover else COLORS.get('BUTTON_NORMAL', (180, 180, 180))
        pygame.draw.rect(screen, header_color, self.header_rect, border_radius=4)
        pygame.draw.rect(screen, COLORS.get('PANEL_BORDER', (100, 100, 100)), self.header_rect, 1, border_radius=4)
        
        # Draw title
        if self.font_manager:
            title_surface = self.font_manager.render_text(self.title, 'BODY', COLORS.get('TEXT_PRIMARY', (0, 0, 0)))
            title_x = self.header_rect.x + 10
            title_y = self.header_rect.y + (self.header_height - title_surface.get_height()) // 2
            screen.blit(title_surface, (title_x, title_y))
        
        # Draw expand/collapse indicator (arrow)
        arrow_size = 8
        arrow_x = self.header_rect.right - 20
        arrow_y = self.header_rect.y + self.header_height // 2
        
        if self.is_expanded:
            # Down arrow (▼)
            points = [
                (arrow_x - arrow_size, arrow_y - arrow_size // 2),
                (arrow_x, arrow_y + arrow_size // 2),
                (arrow_x + arrow_size, arrow_y - arrow_size // 2)
            ]
        else:
            # Right arrow (▶)
            points = [
                (arrow_x - arrow_size // 2, arrow_y - arrow_size),
                (arrow_x + arrow_size // 2, arrow_y),
                (arrow_x - arrow_size // 2, arrow_y + arrow_size)
            ]
        
        pygame.draw.polygon(screen, COLORS.get('TEXT_PRIMARY', (0, 0, 0)), points)
        
        # Draw content area (if expanded)
        if content_visible:
            # Clip drawing to content area
            clip_rect = pygame.Rect(
                self.content_rect.x,
                self.content_rect.y,
                self.content_rect.width,
                self.current_height - self.header_height
            )
            
            # Draw child components
            for component in self.child_components:
                if hasattr(component, 'rect'):
                    # Check if component is within visible area
                    if component.rect.bottom <= clip_rect.bottom:
                        component.draw(screen)
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle events for collapsible panel"""
        if event.type == pygame.MOUSEMOTION:
            mouse_pos = event.pos
            self.is_hover = self.header_rect.collidepoint(mouse_pos)
            return False
        
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos
            if self.header_rect.collidepoint(mouse_pos):
                self.toggle()
                return True
        
        # Pass events to child components if expanded
        if self.is_expanded:
            for component in self.child_components:
                if hasattr(component, 'handle_event'):
                    if component.handle_event(event):
                        return True
        
        return False
    
    def update_child_positions(self, base_y: int) -> None:
        """
        Update positions of child components based on base_y
        
        Args:
            base_y: Base Y coordinate for positioning children
        """
        current_y = base_y
        for component in self.child_components:
            if hasattr(component, 'rect'):
                # Update component position
                component.rect.y = current_y
                # Calculate next position based on component height
                if hasattr(component, 'rect'):
                    current_y = component.rect.bottom + 10  # Add spacing


class ButtonGroup(UIComponent):
    """
    Button group component for mutually exclusive selection
    
    Features:
    - Multiple buttons
    - Single selection
    - Visual feedback
    """
    
    def __init__(
        self,
        rect: pygame.Rect,
        label: str = "",
        options: List[str] = None,
        initial_value: Optional[str] = None,
        font_manager: Optional[Any] = None,
        enabled: bool = True
    ):
        """
        Initialize button group
        
        Args:
            rect: Component rectangle
            label: Label text
            options: List of option strings
            initial_value: Initial selected value
            font_manager: Font manager
            enabled: Whether enabled
        """
        super().__init__(rect, label, font_manager, enabled)
        self.options = options or []
        self.selected_value = initial_value if initial_value in self.options else (self.options[0] if self.options else None)
        self._button_rects: Dict[str, pygame.Rect] = {}
        self._update_button_rects()
    
    def _update_button_rects(self) -> None:
        """Update button rectangles"""
        if not self.options:
            return
        
        button_width = self.rect.width // len(self.options)
        button_height = self.rect.height
        
        for i, option in enumerate(self.options):
            self._button_rects[option] = pygame.Rect(
                self.rect.left + i * button_width,
                self.rect.top,
                button_width,
                button_height
            )
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draw button group"""
        for i, option in enumerate(self.options):
            button_rect = self._button_rects.get(option)
            if not button_rect:
                continue
            
            # Button background
            is_selected = option == self.selected_value
            is_hover = button_rect.collidepoint(pygame.mouse.get_pos()) and self.enabled
            
            if not self.enabled:
                bg_color = COLORS.get('LIGHT_GRAY', (220, 220, 220))
            elif is_selected:
                bg_color = COLORS.get('INFO', (0, 123, 255))
            elif is_hover:
                bg_color = COLORS.get('BUTTON_HOVER', (230, 230, 230))
            else:
                bg_color = COLORS.get('BUTTON_NORMAL', (245, 245, 245))
            
            pygame.draw.rect(screen, bg_color, button_rect)
            
            # Border
            border_color = COLORS.get('INFO', (0, 123, 255)) if is_selected else COLORS.get('PANEL_BORDER', (200, 200, 200))
            pygame.draw.rect(screen, border_color, button_rect, 2)
            
            # Text
            text_color = COLORS.get('TEXT_PRIMARY', (0, 0, 0)) if self.enabled else COLORS.get('GRAY', (120, 120, 120))
            text_surface = self.render_text(option, 'TINY', text_color)
            text_rect = text_surface.get_rect(center=button_rect.center)
            screen.blit(text_surface, text_rect)
        
        # Label
        if self.label:
            label_surface = self.render_text(self.label, 'TINY', COLORS['TEXT_PRIMARY'])
            screen.blit(label_surface, (self.rect.left, self.rect.top - 16))
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle events"""
        if not self.enabled:
            return False
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_pos = event.pos
                for option, button_rect in self._button_rects.items():
                    if button_rect.collidepoint(mouse_pos):
                        self.selected_value = option
                        return True
        
        return False
    
    def get_value(self) -> Optional[str]:
        """Get selected value"""
        return self.selected_value
    
    def set_value(self, value: str) -> None:
        """Set selected value"""
        if value in self.options:
            self.selected_value = value

